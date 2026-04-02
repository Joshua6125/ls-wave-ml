"""Least-Squares algorithm module."""

from .config import LSConfig
from .loss import LSLoss
from .algorithm import LS

__all__ = [
    "LSConfig",
    "LSLoss",
    "LS",
]
