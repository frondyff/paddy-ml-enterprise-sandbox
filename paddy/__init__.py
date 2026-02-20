"""Namespace shim so `python -m paddy...` works from repo root in src-layout."""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg = _pkg_dir.parent / "src" / "paddy"
__path__ = [str(_src_pkg)] if _src_pkg.exists() else []
