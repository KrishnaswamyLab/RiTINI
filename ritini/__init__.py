"""
RiTINI: Regulatory Temporal Interaction Network Inference

A Python package for inferring regulatory temporal interaction networks using
Neural ODEs and Graph Neural Networks.
"""

__version__ = "0.1.0"
__author__ = "Joao Felipe Rocha"
__email__ = "joaofelipe.rocha@yale.edu"

from .ritini import RiTINI
from .data_generation import *

# Make RiTINI the default export for direct usage
__all__ = ["RiTINI"]