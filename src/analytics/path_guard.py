"""Runtime guard keeping investigator-side data access away from ground-truth directories."""

from __future__ import annotations

from pathlib import Path


class ForbiddenDataPathError(PermissionError):
    """Raised when a data loader is pointed at a ground-truth directory."""


def assert_investigator_path(path) -> None:
    """Refuse any path (as given or fully resolved, symlinks included) containing a ground_truth directory."""
    for candidate in (Path(path), Path(path).resolve()):
        for part in candidate.parts:
            if "ground_truth" in part.lower():
                raise ForbiddenDataPathError(
                    f"Refusing to read '{path}': investigator code must never access ground-truth data."
                )
