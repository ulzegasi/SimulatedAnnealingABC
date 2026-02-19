"""io.py."""

import pickle
from pathlib import Path

from .sabc import SABCResult


def save_sabc_result(result: SABCResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_sabc_result(path: str | Path) -> SABCResult:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
