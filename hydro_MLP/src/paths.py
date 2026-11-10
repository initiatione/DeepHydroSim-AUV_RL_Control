"""Path helpers so every script resolves files against hydro_MLP/.

Running ``python src/train.py`` from ``hydro_MLP/`` used to depend on the
current working directory (``../data`` pointed outside the package).  All
scripts now anchor their paths to this file's parent package instead.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(*parts: str) -> str:
    """Join path components onto the hydro_MLP package root."""
    return os.path.join(ROOT, *parts)
