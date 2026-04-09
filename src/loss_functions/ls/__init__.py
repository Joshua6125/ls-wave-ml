"""Least-Squares algorithm module."""

from .config import LSConfig
from .loss import LSLoss
from .method import LS

__all__ = [
    "LSConfig",
    "LSLoss",
    "LS",
]
