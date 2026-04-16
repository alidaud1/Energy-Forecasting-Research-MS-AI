from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union
import zipfile

import pandas as pd

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ZipPath:
    zip_file: Path
    internal_path: str

    def open(self):
        zf = zipfile.ZipFile(self.zip_file)
        return zf, zf.open(self.internal_path)


def ensure_dir(p: PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_from_zip(zip_file: PathLike, internal_path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file inside a zip archive with pandas."""
    zip_file = Path(zip_file)
    with zipfile.ZipFile(zip_file) as zf:
        with zf.open(internal_path) as f:
            return pd.read_csv(f, **kwargs)


def read_csv_from_root(root: PathLike, internal_path: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file from either:
    - a zip archive at `root` (using `internal_path`), or
    - an extracted directory at `root` where `internal_path` is a relative path.

    This keeps the blueprint's internal paths stable:
      - Load:  "Load/Task {k}/L{k}-train.csv"
      - Solar: "Solar/Task {k}/train{k}.csv", ...
    """
    root = Path(root)
    if root.suffix.lower() == ".zip":
        return read_csv_from_zip(root, internal_path, **kwargs)
    return pd.read_csv(root / internal_path, **kwargs)

