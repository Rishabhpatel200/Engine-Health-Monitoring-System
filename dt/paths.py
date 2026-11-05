"""Filesystem locations that do not depend on the current working directory."""
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("AERODT_DATA_DIR") or os.path.join(ROOT, "data")


def resolve(path: str) -> str:
    """Resolve a project-relative path (e.g. 'data/models.json') against ROOT."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def writable_dir(preferred: str) -> str:
    """Return `preferred` if it can be written, else a temp directory.

    Many hosts mount the application read-only (or give it an ephemeral disk);
    the flight recorder must degrade rather than stop the service starting.
    """
    try:
        os.makedirs(preferred, exist_ok=True)
        probe = os.path.join(preferred, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return preferred
    except OSError:
        fallback = os.path.join(tempfile.gettempdir(), "aerodt")
        os.makedirs(fallback, exist_ok=True)
        return fallback
