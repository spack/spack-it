import io
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from podman import PodmanClient


class SpackError(Exception):
    def __init__(self, out: str):
        super().__init__(out)
        self.output = out


# LLNL's network re-signs outbound HTTPS certs with its own internal CA
# (cspca.llnl.gov) rather than blocking them outright. The host trusts that CA
# already (see HOST_CA_BUNDLE below); the container image (built from a generic
# public base image) doesn't, which breaks spack's buildcache index fetch --
# a urllib/ssl code path controlled by config:ssl_certs, not config:verify_ssl.
# Copying the host's already-trusted bundle in on every container start (rather
# than baking a static copy into the image) means this stays correct even if
# the host's CA bundle is later rotated, and needs no image rebuild.
HOST_CA_BUNDLE = Path(os.getenv("HOST_CA_BUNDLE", "/etc/pki/tls/certs/ca-bundle.crt"))
CONTAINER_CA_BUNDLE = "/etc/llnl-ca-bundle.crt"


RESERVED_NAMES_ONLY_LOWERCASE = frozenset(
    (
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    )
)


class BuilderContainer:
    def __init__(self, image="builder", namespace="experiment", log: Callable[[str], None] = print):
        self.image = image
        self.socket = os.getenv("PODMAN_SOCKET")
        self.client = None
        self.container = None
        # TODO add comment
        self.pkgs_path = "/home/spack/experiment/spack_repo/experiment/packages"
        self.namespace = namespace
        self._log = log

    def __enter__(self):
        self.client = PodmanClient(base_url=f"unix://{self.socket}")
        self.container = self.client.containers.run(
            self.image,
            command=["sleep", "infinity"],
            detach=True,
        )
        self._install_ca_bundle()
        return self

    def _install_ca_bundle(self):
        """copy the host's trusted CA bundle into the container and point spack
        at it, so buildcache index fetches survive LLNL's TLS interception --
        see the HOST_CA_BUNDLE comment above. No-op (with a log line) if the
        host doesn't have a bundle at that path, e.g. outside LLNL's network."""
        if not HOST_CA_BUNDLE.exists():
            self._log(
                f"note: {HOST_CA_BUNDLE} not found on host, skipping CA bundle install "
                "(set HOST_CA_BUNDLE if it lives elsewhere on this system)"
            )
            return
        self._write_file(CONTAINER_CA_BUNDLE, HOST_CA_BUNDLE.read_text())
        self.exec(f"spack config add config:ssl_certs:{CONTAINER_CA_BUNDLE}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.container:
            try:
                self.container.kill()
            except Exception:
                pass
            try:
                self.container.remove(force=True)
            except Exception:
                pass
        if self.client:
            self.client.close()

    def exec(self, cmd, workdir=None):
        self._log(cmd)
        code, out = self.container.exec_run(cmd, demux=True, workdir=workdir)
        stdout, stderr = out
        return (
            code,
            (stderr or b"").decode("utf-8", "replace").rstrip(),
            (stdout or b"").decode("utf-8", "replace").rstrip(),
        )

    def _raise_if_failed(self, code: int, out: str):
        if code != 0:
            raise SpackError(out)

    def _write_file(self, dest: str, content: str):
        """Write string content into a file inside the container at `dest`,
        creating any missing parent directories automatically.
        """
        # Ensure directory exists
        parent_dir = os.path.dirname(dest) or "/"
        self.container.exec_run(f"mkdir -p {parent_dir}")

        # Pack file content into a tar stream
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            arcname = os.path.basename(dest)
            data = content.encode()
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)

        # Put into container
        self.container.put_archive(parent_dir, buf.read())

    def write_pkg(self, name: str, recipe: str) -> str:
        normal = name.lower().replace("-", "_")
        if re.match(r"^[0-9]", normal) or normal in RESERVED_NAMES_ONLY_LOWERCASE:
            normal = f"_{normal}"

        dest = os.path.join(self.pkgs_path, normal, "package.py")
        self._write_file(dest, recipe)
        return normal

    def load_pkg(self, pkg: str):
        cmd = f"spack list {self.namespace}.{pkg}"
        code, stderr, _ = self.exec(cmd)
        self._raise_if_failed(code, stderr)

    def concretize_pkg(self, pkg: str):
        cmd = f"spack spec {self.namespace}.{pkg}"
        code, stderr, _ = self.exec(cmd)
        self._raise_if_failed(code, stderr)

    def install_pkg(self, pkg: str):
        cmd = f"spack install -y --concurrent-packages 16 --use-buildcache package:never,dependencies:auto {self.namespace}.{pkg}"
        code, stderr, _ = self.exec(cmd)
        self._raise_if_failed(code, stderr)

    def test_pkg(self, pkg: str):
        cmd = f"spack test run --fail-fast {self.namespace}.{pkg}"
        code, stderr, _ = self.exec(cmd)
        self._raise_if_failed(code, stderr)

    def audit_pkg(self, pkg: str):
        cmd = f"spack audit packages {self.namespace}.{pkg}"
        code, stderr, stdout = self.exec(cmd)
        if not stderr and not stdout:
            return
        return f"""{stdout}

        {stderr}
        """

    def deps_score(self, pkg: str) -> str:
        cmd = f"spack python scores.py --pkg_name {pkg} --score deps"
        code, stderr, stdout = self.exec(cmd, workdir="/opt/spack-it")
        self._raise_if_failed(code, stderr)
        return float(stdout)

    def cmake_args_score(self, pkg: str) -> str:
        cmd = f"spack python scores.py --pkg_name {pkg} --score cmake"
        code, stderr, stdout = self.exec(cmd, workdir="/opt/spack-it")
        self._raise_if_failed(code, stderr)
        score, extra = stdout.split(",")
        return float(score), int(extra)


@dataclass
class Stage:
    name: str
    action: Callable[[], None]
