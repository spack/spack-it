# rag0_index.py
# deps:
#   pip install torch sentence-transformers faiss-cpu numpy

import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import faiss  # faiss-cpu
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---- CUDA allocator hint (set before torch/cuda init) ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

EMBED_MAX_SEQ_LEN = int(os.getenv("SPACK_RAG_MAX_SEQ_LEN", "512"))
SINGLE_GPU_START_BATCH = int(os.getenv("SPACK_RAG_SINGLE_GPU_BATCH", "64"))
MULTI_GPU_START_BATCH = int(os.getenv("SPACK_RAG_MULTI_GPU_BATCH", "64"))
USE_FP16_SINGLE_GPU = os.getenv("SPACK_RAG_FP16", "1") != "0"


def have_gpu() -> bool:
    try:
        return torch.cuda.device_count() > 0
    except Exception:
        return False


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


_SECTION_PATTERNS = [
    ("variants", re.compile(r"^\s*variant\s*\(", re.IGNORECASE)),
    ("depends", re.compile(r"^\s*depends_on\s*\(", re.IGNORECASE)),
    ("install", re.compile(r"^\s*def\s+install\s*\(", re.IGNORECASE)),
    ("check", re.compile(r"^\s*def\s+check\s*\(", re.IGNORECASE)),
    (
        "build",
        re.compile(
            r"^\s*def\s+build_[a-zA-Z_]*\s*\(|^\s*@property\s*\n\s*def\s+build_targets",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    ("class", re.compile(r"^\s*class\s+\w+\(", re.IGNORECASE)),
]


def chunk_recipe(recipe: str, max_lines_per_chunk: int = None) -> List[Dict[str, str]]:
    lines = recipe.splitlines()
    chunks, lab, buf = [], "other", []

    def flush(l, b):
        if not b:
            return
        if max_lines_per_chunk and max_lines_per_chunk > 0:
            b = b[:max_lines_per_chunk]
        t = "\n".join(b).strip()
        if t:
            chunks.append({"id": l, "text": t})

    def label_for(line):
        for L, pat in _SECTION_PATTERNS:
            if pat.search(line):
                return L
        return "other"

    for line in lines:
        newL = label_for(line)
        if newL != lab and buf:
            flush(lab, buf)
            buf = []
        lab = newL
        buf.append(line)
    flush(lab, buf)
    return chunks


@dataclass
class Corpus:
    names: List[str]
    cards: List[str]
    chunks: List[str]
    chunk_pkg_idx: np.ndarray


def build_corpus(
    packages_by_name: Dict[str, Any], max_lines_per_chunk: int = None
) -> Corpus:
    pkgs = [
        v for _, v in sorted(packages_by_name.items(), key=lambda kv: kv[0].lower())
    ]
    names = [extract_name(p) for p in pkgs]
    cards = [feature_card(p) for p in pkgs]

    chunks, owners = [], []
    for i, p in enumerate(pkgs):
        for ch in chunk_recipe(
            extract_recipe(p), max_lines_per_chunk=max_lines_per_chunk
        ):
            chunks.append(ch["text"])
            owners.append(i)
    if not chunks:
        chunks, owners = [""], [0]
    return Corpus(
        names=names,
        cards=cards,
        chunks=chunks,
        chunk_pkg_idx=np.array(owners, dtype=np.int32),
    )


@dataclass
class RAG0Index:
    model_name: str
    dim: int
    card_index: faiss.Index
    card_vecs: np.ndarray
    chunk_index: faiss.Index
    chunk_vecs: np.ndarray
    corpus: Corpus


def _load_model(model_name: str):
    model = SentenceTransformer(model_name)
    try:
        if EMBED_MAX_SEQ_LEN:
            model.max_seq_length = min(
                getattr(model, "max_seq_length", EMBED_MAX_SEQ_LEN), EMBED_MAX_SEQ_LEN
            )
    except Exception:
        pass
    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and USE_FP16_SINGLE_GPU
    ):
        try:
            model = model.to(torch.device("cuda"))
            model.half()
        except Exception:
            pass
    return model


def _encode_single_device(model, texts: List[str], start_bs: int) -> np.ndarray:
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
    raise last_err if last_err else RuntimeError("Failed to encode on single device.")


def _encode_multi_gpu(model, texts: List[str], start_bs: int) -> np.ndarray:
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
                    continue
                raise
        if last_err:
            return _encode_single_device(model, texts, start_bs=32)
        raise RuntimeError("encode_multi_process failed unexpectedly.")
    finally:
        try:
            model.stop_multi_process_pool(pool)
        except Exception:
            pass


def _encode(model, texts: List[str], batch_size: int = None) -> np.ndarray:
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


def _faiss_index_cpu(dim: int, vecs: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatIP(dim)  # cosine via inner product on normalized vectors
    idx.add(vecs)
    return idx


def build_index(
    packages_by_name: Dict[str, Any],
    model_name: str,
    max_lines_per_chunk: int = None,
) -> RAG0Index:
    corpus = build_corpus(packages_by_name, max_lines_per_chunk=max_lines_per_chunk)
    model = _load_model(model_name)

    cards_for_emb = [f"passage: {t}" for t in corpus.cards]
    chunks_for_emb = [f"passage: {t}" for t in corpus.chunks]

    card_vecs = _encode(model, cards_for_emb, batch_size=None)
    chunk_vecs = _encode(model, chunks_for_emb, batch_size=None)
    d = card_vecs.shape[1]

    card_index = _faiss_index_cpu(d, card_vecs)
    chunk_index = _faiss_index_cpu(d, chunk_vecs)

    return RAG0Index(
        model_name, d, card_index, card_vecs, chunk_index, chunk_vecs, corpus
    )


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


def save_corpus(corpus: Corpus, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "names.pkl"), "wb") as f:
        pickle.dump(corpus.names, f)
    with open(os.path.join(cache_dir, "cards.pkl"), "wb") as f:
        pickle.dump(corpus.cards, f)
    with open(os.path.join(cache_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(corpus.chunks, f)
    np.save(os.path.join(cache_dir, "chunk_pkg_idx.npy"), corpus.chunk_pkg_idx)


def load_corpus(cache_dir: str) -> Corpus:
    with open(os.path.join(cache_dir, "names.pkl"), "rb") as f:
        names = pickle.load(f)
    with open(os.path.join(cache_dir, "cards.pkl"), "rb") as f:
        cards = pickle.load(f)
    with open(os.path.join(cache_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    chunk_pkg_idx = np.load(os.path.join(cache_dir, "chunk_pkg_idx.npy"))
    return Corpus(names, cards, chunks, chunk_pkg_idx)


def save_embs_and_indexes(
    cache_dir: str, card_vecs, chunk_vecs, card_index, chunk_index
):
    np.save(os.path.join(cache_dir, "card_vecs.npy"), card_vecs)
    np.save(os.path.join(cache_dir, "chunk_vecs.npy"), chunk_vecs)
    faiss.write_index(card_index, os.path.join(cache_dir, "card_index.faiss"))
    faiss.write_index(chunk_index, os.path.join(cache_dir, "chunk_index.faiss"))


def load_embs_and_indexes(cache_dir: str):
    card_vecs = np.load(os.path.join(cache_dir, "card_vecs.npy"))
    chunk_vecs = np.load(os.path.join(cache_dir, "chunk_vecs.npy"))
    card_index = faiss.read_index(os.path.join(cache_dir, "card_index.faiss"))
    chunk_index = faiss.read_index(os.path.join(cache_dir, "chunk_index.faiss"))
    return card_vecs, chunk_vecs, card_index, chunk_index


def build_index_cached(
    packages_by_name: Dict[str, Any],
    model_name: str = "intfloat/e5-base-v2",
    max_lines_per_chunk: int = None,
    cache_root: str = ".cache/spackrag",
    source_path: str = "",
) -> RAG0Index:
    h = dataset_fingerprint(packages_by_name, source_path=source_path)
    cache_dir = os.path.join(
        cache_root, h, model_name.replace("/", "_"), f"lines{max_lines_per_chunk or 0}"
    )
    # Try load
    try:
        corpus = load_corpus(cache_dir)
        card_vecs, chunk_vecs, card_index, chunk_index = load_embs_and_indexes(
            cache_dir
        )
        return RAG0Index(
            model_name,
            card_vecs.shape[1],
            card_index,
            card_vecs,
            chunk_index,
            chunk_vecs,
            corpus,
        )
    except Exception:
        pass  # cache miss → build fresh

    idx = build_index(
        packages_by_name, model_name=model_name, max_lines_per_chunk=max_lines_per_chunk
    )

    save_corpus(idx.corpus, cache_dir)
    save_embs_and_indexes(
        cache_dir, idx.card_vecs, idx.chunk_vecs, idx.card_index, idx.chunk_index
    )
    return idx


if __name__ == "__main__":
    data_path = os.path.join("data", "packages.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}.")
    with open(data_path, "rb") as f:
        packages_by_name = pickle.load(f)
    idx = build_index_cached(
        packages_by_name,
        model_name="nomic-ai/nomic-embed-code",
        max_lines_per_chunk=20,
        cache_root="/usr/workspace/user/spackrag_cache",
        source_path=data_path,
    )
    print("Index built and cached.")
