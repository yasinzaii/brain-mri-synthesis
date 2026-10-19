from __future__ import annotations

import hashlib
from pathlib import Path


def strip_nii_gz(name: str) -> tuple[str, str]:
    if name.endswith(".nii.gz"):
        return name[:-7], ".nii.gz"
    p = Path(name)
    return p.stem, p.suffix


def add_postfix(rel_in: Path, postfix: str, *, extension: str | None = None) -> Path:
    """Add postfix before extension, preserving .nii.gz unless extension is set."""
    base, ext = strip_nii_gz(rel_in.name)
    postfix = postfix.lstrip("_")
    out_ext = ext if extension is None else extension
    if out_ext and not out_ext.startswith("."):
        out_ext = f".{out_ext}"
    return rel_in.with_name(f"{base}_{postfix}{out_ext}")


def ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty directory: {path}. "
            "Set overwrite=true to allow."
        )


def relpath_from_abs(abs_path: str | Path, *, root: Path) -> Path:
    p = Path(abs_path)
    try:
        return p.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Path '{p}' is not under root '{root}'") from e


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
