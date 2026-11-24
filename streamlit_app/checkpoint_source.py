from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import streamlit as st

from checkpoint import MAX_CHECKPOINT_BYTES
from config import GENERATOR_CHECKPOINT_PATH, GENERATOR_CHECKPOINT_URL

_TOO_LARGE = "The checkpoint is too large. The maximum is 512 MB."
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


@st.cache_resource(show_spinner=False)
def _download_checkpoint(url: str) -> str:
    request = Request(url, headers={"User-Agent": "SolFlow-generator/1.0"})
    with urlopen(request, timeout=30) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise ValueError("The checkpoint URL must remain HTTPS after redirects.")
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_CHECKPOINT_BYTES:
            raise ValueError(_TOO_LARGE)
        payload = response.read(MAX_CHECKPOINT_BYTES + 1)
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError(_TOO_LARGE)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"solflow_geometry_{digest}.pt"
    path.write_bytes(payload)
    return str(path)


def list_checkpoints() -> list[str]:
    # .pt/.pth files in the checkpoints folder, as paths relative to it.
    if not CHECKPOINT_DIR.is_dir():
        return []
    files = [p for p in CHECKPOINT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".pth"}]
    return sorted(p.relative_to(CHECKPOINT_DIR).as_posix() for p in files)


def default_checkpoint_index(options: list[str]) -> int:
    # Prefer the deployment model (flow_general), then fold 0, then the first file.
    for key in ("general", "fold0", "fold_0"):
        for i, name in enumerate(options):
            if key in name.lower():
                return i
    return 0


def resolve_checkpoint_path(selected: str | None = None) -> Path:
    if selected:
        path = (CHECKPOINT_DIR / selected).resolve()
        if CHECKPOINT_DIR not in path.parents:
            raise ValueError("Invalid checkpoint selection.")
        if not path.is_file():
            raise ValueError(f"Checkpoint was not found: {path}")
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError(_TOO_LARGE)
        return path
    parsed = urlparse(GENERATOR_CHECKPOINT_URL)
    if GENERATOR_CHECKPOINT_URL and parsed.scheme == "https" and parsed.netloc:
        return Path(_download_checkpoint(GENERATOR_CHECKPOINT_URL))
    if GENERATOR_CHECKPOINT_PATH:
        path = Path(GENERATOR_CHECKPOINT_PATH).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        path = path.resolve()
        if path.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("GENERATOR_CHECKPOINT_PATH must point to a .pt or .pth file.")
        if not path.is_file():
            raise ValueError(f"Local generator checkpoint was not found: {path}")
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError(_TOO_LARGE)
        return path
    raise ValueError("No checkpoint found. Put a .pt file in the checkpoints folder, "
                     "or set GENERATOR_CHECKPOINT_URL or GENERATOR_CHECKPOINT_PATH in config.py.")
