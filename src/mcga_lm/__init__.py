"""MCGA-LM: Multimodal Context Graph-Augmented Language Model.

Official implementation of Al-Nefaie, Wadood, Aldhyani, Uddin & Saeed,
"MCGA-LM: Multimodal Context Graph-Augmented Language Model for Adaptive Intent
Reconstruction in Assistive Communication".

Every experiment here runs on synthetic personas, as the paper's evaluation
does. The default language backend is weight-free so the pipeline runs without
model weights. See the Reproduction status section of README.md for what the
default configuration does and does not reproduce.
"""

from .config import Config
from .seed import set_seed

__version__ = "0.1.0"
__all__ = ["Config", "set_seed"]
