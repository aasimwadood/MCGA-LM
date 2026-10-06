"""Training: encoder pre-training and per-user personalisation (paper Sec. 3.7).
"""

from .personalise import TrainingReport, calibrate_thresholds, train
from .pretrain import pretrain

__all__ = ["pretrain", "train", "TrainingReport", "calibrate_thresholds"]
