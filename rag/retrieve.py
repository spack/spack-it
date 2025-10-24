# rag0_retrieve.py
# deps:
#   pip install torch sentence-transformers faiss-cpu numpy

import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Keep allocator hint; harmless here too
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Tunables for query encoding only
EMBED_MAX_SEQ_LEN = int(os.getenv("SPACK_RAG_MAX_SEQ_LEN", "512"))
SINGLE_GPU_START_BATCH = int(os.getenv("SPACK_RAG_SINGLE_GPU_BATCH", "64"))
MULTI_GPU_START_BATCH = int(os.getenv("SPACK_RAG_MULTI_GPU_BATCH", "64"))
USE_FP16_SINGLE_GPU = os.getenv("SPACK_RAG_FP16", "1") != "0"


def _get(obj, name, default=None):
    return getattr(
        obj, name, obj.get(name, default) if isinstance(obj, dict) else default
    )


def _iter(x) -> Iterable:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return x
    return [x]


def truncate_lines(text: str, max_lines: int = 12) -> str:
    if not max_lines or max_lines <= 0:
        return text
    return "\n".join(text.splitlines()[:max_lines])


def extract_name(pkg) -> str:
    return str(_get(pkg, "name", "")).strip()


def extract_dependencies(pkg) -> List[Tuple[str, str]]:
    out = []
    for d in _iter(_get(pkg, "dependencies", [])):
        dn = (_get(d, "pkg_name", "") or "").lower().strip()
        for t in _get(d, "types", tuple()) or ():
            out.append((dn, str(t).lower()))
    return out


def extract_variants(pkg) -> List[Tuple[str, str]]:
    vs = []
    for v in _iter(_get(pkg, "variants", [])):
        n = (_get(v, "name", "") or "").lower()
        default = _get(v, "default", None)
        if isinstance(default, bool):
            val = "true" if default else "false"
        elif default is None:
            vals = [_get(x, "value", None) for x in _iter(_get(v, "values", []))]
            val = str(vals[0]).lower() if vals and vals[0] is not None else "none"
        else:
            val = str(default).lower()
        vs.append((n, val))
    return vs


def extract_build_systems(pkg) -> List[str]:
    bs = _get(pkg, "build_systems", set())
    return sorted([str(x).lower() for x in _iter(bs)])


def extract_recipe(pkg) -> str:
    return _get(pkg, "recipe", "") or ""


def feature_card(pkg) -> str:
    nm = extract_name(pkg).lower()
    bs = extract_build_systems(pkg)
    deps = extract_dependencies(pkg)
    vars_ = extract_variants(pkg)

    by_type = {}
    for dn, dt in deps:
        by_type.setdefault(dt, []).append(dn)

    deps_txt = " ".join(
        f"dep_{t}:" + ",".join(sorted(set(v))) for t, v in sorted(by_type.items())
    )
    vars_txt = " ".join(f"var_{n}=={val}" for n, val in sorted(set(vars_)))
    bs_txt = " ".join(f"buildsys:{b}" for b in bs)

    return " | ".join(x for x in [f"name:{nm}", bs_txt, deps_txt, vars_txt] if x)


@dataclass
class Corpus:
    names: List[str]
    cards: List[str]
    chunks: List[str]
    chunk_pkg_idx: np.ndarray


@dataclass
class RAG0Index:
    model_name: str
    dim: int
    card_index: faiss.Index
    card_vecs: np.ndarray
    chunk_index: faiss.Index
    chunk_vecs: np.ndarray
    corpus: Corpus


def load_corpus(cache_dir: str) -> Corpus:
    with open(os.path.join(cache_dir, "names.pkl"), "rb") as f:
        names = pickle.load(f)
    with open(os.path.join(cache_dir, "cards.pkl"), "rb") as f:
        cards = pickle.load(f)
    with open(os.path.join(cache_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    chunk_pkg_idx = np.load(os.path.join(cache_dir, "chunk_pkg_idx.npy"))
    return Corpus(names, cards, chunks, chunk_pkg_idx)


def load_embs_and_indexes(cache_dir: str):
    card_vecs = np.load(os.path.join(cache_dir, "card_vecs.npy"))
    chunk_vecs = np.load(os.path.join(cache_dir, "chunk_vecs.npy"))
    card_index = faiss.read_index(os.path.join(cache_dir, "card_index.faiss"))
    chunk_index = faiss.read_index(os.path.join(cache_dir, "chunk_index.faiss"))
    return card_vecs, chunk_vecs, card_index, chunk_index


def dataset_fingerprint(packages_by_name: Dict[str, Any], source_path: str = "") -> str:
    items = [(k, packages_by_name[k]) for k in sorted(packages_by_name)]
    m = hashlib.sha1()
    if source_path and os.path.exists(source_path):
        m.update(source_path.encode())
        m.update(str(os.path.getmtime(source_path)).encode())
    for k, p in items:
        card = feature_card(p)
        rec = extract_recipe(p)
        m.update(k.encode())
        m.update(card.encode())
        m.update(str(len(rec)).encode())
    return m.hexdigest()[:12]


def load_index_from_cache(
    cache_root: str,
    model_name: str,
    max_lines_per_chunk: int,
    *,
    cache_dir: str = None,
    packages_by_name: Dict[str, Any] = None,
    source_path: str = "",
) -> RAG0Index:
    """
    Load an index strictly from cache. If cache_dir is provided, use it directly.
    Otherwise, compute the cache path using the SAME scheme as the indexer
    (requires packages_by_name to compute the fingerprint).
    """
    if cache_dir is None:
        if packages_by_name is None:
            raise ValueError(
                "Either provide cache_dir explicitly, or pass packages_by_name to compute fingerprint."
            )
        h = dataset_fingerprint(packages_by_name, source_path=source_path)
        cache_dir = os.path.join(
            cache_root,
            h,
            model_name.replace("/", "_"),
            f"lines{max_lines_per_chunk or 0}",
        )

    corpus = load_corpus(cache_dir)
    card_vecs, chunk_vecs, card_index, chunk_index = load_embs_and_indexes(cache_dir)
    d = int(card_vecs.shape[1])
    return RAG0Index(
        model_name, d, card_index, card_vecs, chunk_index, chunk_vecs, corpus
    )


_DEP_RE = re.compile(r"[A-Za-z0-9_\-\.]+")
_VAR_POS = re.compile(r"\+([A-Za-z0-9_\-]+)")
_VAR_NEG = re.compile(r"~([A-Za-z0-9_\-]+)")
_SPEC_EQ = re.compile(r"([A-Za-z0-9_\-]+)\s*=\s*([A-Za-z0-9_\-\.]+)")


def _tokens_from_text(text: str) -> Tuple[List[str], List[str]]:
    deps, varflags = [], []
    varflags += [f"{m.group(1).lower()}==true" for m in _VAR_POS.finditer(text)]
    varflags += [f"{m.group(1).lower()}==false" for m in _VAR_NEG.finditer(text)]
    varflags += [f"{k.lower()}=={v.lower()}" for k, v in _SPEC_EQ.findall(text)]
    for w in _DEP_RE.findall(text):
        wl = w.lower()
        if wl in {
            "dep",
            "deps",
            "depends",
            "depends_on",
            "variant",
            "variants",
            "buildsys",
            "build_system",
            "when",
        }:
            continue
        if wl in {"true", "false", "and", "or", "not"}:
            continue
        deps.append(wl)
    return deps, varflags


def query_text_from_input(q: Union[str, List[str], Dict[str, Any], Any]) -> str:
    deps, varpairs, bsys, name_hint = [], [], [], ""
    if isinstance(q, str):
        d, v = _tokens_from_text(q)
        deps += d
        varpairs += v
    elif isinstance(q, list):
        d, v = _tokens_from_text(" ".join(map(str, q)))
        deps += d
        varpairs += v
    elif isinstance(q, dict):
        name_hint = str(q.get("name", ""))
        for d in _iter(q.get("deps", [])):
            if isinstance(d, str):
                deps.append(d.lower())
            else:
                deps.append(str(d.get("name", "")).lower())
        for v in _iter(q.get("variants", [])):
            if isinstance(v, str):
                d2, v2 = _tokens_from_text(v)
                deps += d2
                varpairs += v2
            else:
                n = str(v.get("name", "")).lower()
                val = v.get("default", None)
                if isinstance(val, bool):
                    val = "true" if val else "false"
                elif val is None:
                    val = "none"
                else:
                    val = str(val).lower()
                varpairs.append(f"{n}=={val}")
        for b in _iter(q.get("build_systems", [])):
            bsys.append(str(b).lower())
    else:
        # Package-like
        name_hint = extract_name(q)
        deps += [dn for dn, _ in extract_dependencies(q)]
        varpairs += [f"{n}=={v}" for n, v in extract_variants(q)]
        bsys += extract_build_systems(q)

    toks = []
    if name_hint:
        toks.append(f"name:{name_hint.lower()}")
    toks += [f"dep:{d}" for d in deps]
    toks += [f"var:{p}" for p in varpairs]
    toks += [f"buildsys:{b}" for b in bsys]
    if not toks:
        toks = ["spack package similarity", "depends_on", "variant"]
    return "query: " + " ".join(toks)


def _load_model(model_name: str):
    """
    Create the SentenceTransformer model and cap sequence length to reduce memory.
    For single-GPU runs, we optionally cast to FP16 to save VRAM.
    """
    model = SentenceTransformer(model_name)
    try:
        if EMBED_MAX_SEQ_LEN:
            model.max_seq_length = min(
                getattr(model, "max_seq_length", EMBED_MAX_SEQ_LEN), EMBED_MAX_SEQ_LEN
            )
    except Exception:
        pass

    # Only cast to half for the single-GPU path (multi-process pool manages its own replicas)
    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and USE_FP16_SINGLE_GPU
    ):
        try:
            model = model.to(torch.device("cuda"))
            model.half()
        except Exception:
            # If half precision isn't supported, keep float32
            pass
    return model


def _encode_single_device(model, texts: List[str], start_bs: int) -> np.ndarray:
    """Encode with automatic batch-size backoff on CUDA OOM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backoff = [start_bs, 32, 16, 8, 4, 2, 1]
    last_err = None
    for bs in backoff:
        try:
            embs = model.encode(
                texts,
                batch_size=bs,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                device=device,
            )
            return embs.astype(np.float32, copy=False)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cuda out of memory" in msg or "cublas" in msg:
                last_err = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    # If we got here, all attempts failed
    raise last_err if last_err else RuntimeError("Failed to encode on single device.")


def _encode_multi_gpu(model, texts: List[str], start_bs: int) -> np.ndarray:
    """
    Multi-GPU using SentenceTransformers' worker pool.
    Uses conservative per-device batch size; if it still OOMs, falls back to single device.
    """
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    pool = model.start_multi_process_pool(target_devices=devices)
    bs_backoff = [start_bs, 32, 16, 8, 4, 2, 1]
    try:
        last_err = None
        for bs in bs_backoff:
            try:
                embs = model.encode_multi_process(
                    texts,
                    pool,
                    batch_size=bs,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
                return np.asarray(embs, dtype=np.float32)
            except RuntimeError as e:
                msg = str(e).lower()
                if "cuda out of memory" in msg:
                    last_err = e
                    # pool stays alive across attempts
                    continue
                raise
        # All multi-GPU attempts failed → fallback to single device
        if last_err:
            return _encode_single_device(model, texts, start_bs=32)
        raise RuntimeError("encode_multi_process failed unexpectedly.")
    finally:
        try:
            model.stop_multi_process_pool(pool)
        except Exception:
            pass


DISABLE_MP_FOR_QUERIES = os.getenv("RAG_DISABLE_MP", "1") != "0"


def _encode(model, texts: List[str], batch_size: int = None) -> np.ndarray:
    """
    Encode with:
      - Multi-GPU pool if >=2 GPUs (with backoff)
      - Otherwise single-device with backoff
    """
    if DISABLE_MP_FOR_QUERIES or len(texts) < 32:
        return _encode_single_device(model, texts, start_bs=batch_size or 32)
    if batch_size is None:
        batch_size = (
            MULTI_GPU_START_BATCH
            if torch.cuda.device_count() >= 2
            else SINGLE_GPU_START_BATCH
        )
    if torch.cuda.device_count() >= 2:
        return _encode_multi_gpu(model, texts, start_bs=batch_size)
    else:
        return _encode_single_device(model, texts, start_bs=batch_size)


# Retrieval (RAG-0; just chunks)
def retrieve_chunks(
    index: RAG0Index,
    query_input: Union[str, List[str], Dict[str, Any], Any],
    top_k_packages: int = 10,
    chunks_per_pkg: int = 2,
    exclude_package_name: str = "",
    output_line_cap: int = None,
) -> List[Dict[str, Any]]:
    q_model = _load_model(index.model_name)
    # disabling this for now bc we want to form the query ourselves.
    # THE QUERY_INPUT MUST START WITH QUERY:
    # qtext = query_text_from_input(query_input)
    qvec = _encode(q_model, [query_input], batch_size=32)  # normalized

    sims = (qvec @ index.card_vecs.T).ravel()
    order = np.argsort(-sims)

    results, seen = [], set()
    for i in order:
        pkg_name = index.corpus.names[i]
        if exclude_package_name and pkg_name.lower() == exclude_package_name.lower():
            continue
        if pkg_name in seen:
            continue
        seen.add(pkg_name)

        mask = index.corpus.chunk_pkg_idx == i
        if not mask.any():
            results.append({"package": pkg_name, "score": float(sims[i]), "chunks": []})
        else:
            pkg_chunk_vecs = index.chunk_vecs[mask]
            csims = (qvec @ pkg_chunk_vecs.T).ravel()
            loc = np.argsort(-csims)[:chunks_per_pkg]
            global_ids = np.where(mask)[0][loc]
            texts = [index.corpus.chunks[g] for g in global_ids]
            if output_line_cap and output_line_cap > 0:
                texts = [truncate_lines(t, output_line_cap) for t in texts]
            results.append(
                {"package": pkg_name, "score": float(sims[i]), "chunks": texts}
            )

        if len(results) >= top_k_packages:
            break
    return results
