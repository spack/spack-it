import atexit
import select
import shutil
import sys
import threading
from collections import deque

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX fallback
    termios = None
    tty = None


class ProgressUI:
    """
    Pins a single status line ("[7/40] pkg=foo attempt=2/5 stage=concretize")
    at the bottom of the terminal while a run is in progress, instead of the
    raw print() stream (spack commands, LLM calls, per-attempt errors) that
    normally scrolls past. Press 'v' to toggle that raw stream back on above
    the status line; press it again to hide it.

    Falls back to plain print() with no in-place redraw and no keypress
    listener when stdout/stdin isn't a real terminal (piped to a file, etc.),
    since neither in-place redraw nor a keypress toggle means anything there.
    """

    def __init__(self, total: int, max_attempts: int, log_buffer_len: int = 200):
        self.total = total
        self.max_attempts = max_attempts
        self.completed = 0
        self.current_pkg = None
        self.attempt = 0
        self.stage = None
        self.verbose = False

        self._log_buffer = deque(maxlen=log_buffer_len)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._key_thread = None
        self._old_termios = None

        self._interactive = sys.stdout.isatty() and sys.stdin.isatty() and termios is not None
        if self._interactive:
            try:
                fd = sys.stdin.fileno()
                self._old_termios = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except (termios.error, ValueError):
                self._interactive = False

        if self._interactive:
            self._key_thread = threading.Thread(target=self._listen_for_keys, daemon=True)
            self._key_thread.start()
            atexit.register(self.close)

    # ---- status updates, called from the generation pipeline ----

    def start_pkg(self, pkg_name: str):
        with self._lock:
            self.current_pkg = pkg_name
            self.attempt = 0
            self.stage = None
        self._redraw()

    def set_attempt(self, attempt_num: int, error: str = None):
        with self._lock:
            self.attempt = attempt_num
        self.log(f"attempt={attempt_num}, prev_error={error}")

    def set_stage(self, stage: str):
        with self._lock:
            self.stage = stage
        self._redraw()

    def finish_pkg(self):
        with self._lock:
            self.completed += 1
        self._redraw()

    # ---- logging: replaces the raw print() calls throughout the pipeline ----

    def log(self, msg: str):
        self._log_buffer.append(msg)
        if not self._interactive:
            print(msg)
            return
        if self.verbose:
            self._write_line(msg)
        self._redraw()

    # ---- rendering ----

    def _status_text(self) -> str:
        width = shutil.get_terminal_size((80, 20)).columns
        pkg = self.current_pkg or "-"
        stage = self.stage or "-"
        mode = "logs: on (press v to hide)" if self.verbose else "press v for logs"
        text = (
            f"[{self.completed}/{self.total}] pkg={pkg} "
            f"attempt={self.attempt}/{self.max_attempts} stage={stage} | {mode}"
        )
        return text[: max(width - 1, 0)]

    def _write_line(self, msg: str):
        # erase the pinned status line, print a real log line above it; the
        # next _redraw() repins the status line at the new bottom
        sys.stdout.write("\r\x1b[K" + msg + "\n")
        sys.stdout.flush()

    def _redraw(self):
        if not self._interactive:
            return
        with self._lock:
            text = self._status_text()
        sys.stdout.write("\r\x1b[K" + text)
        sys.stdout.flush()

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()
            if self._old_termios is not None:
                try:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
                except termios.error:
                    pass

    # ---- keypress listener (background thread) ----

    def _listen_for_keys(self):
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch.lower() == "v":
                self.verbose = not self.verbose
                if self.verbose:
                    # replay recent history so toggling on doesn't lose context
                    for line in list(self._log_buffer):
                        self._write_line(line)
                self._redraw()
