"""
HierHGT-DTI model package (packed-only backend).
"""

from .hierhgt_dti_dataset import HierHGTDTIDataset, hierhgt_dti_collate_fn
from .hierhgt_dti_model import HierHGTDTIModel
from .encoders import AtomEncoder, ResidueEncoder, SharedNodeStem
from .packed_hgt_layers import PackedHGTConv, PackedJointHGT, is_torch_scatter_available

__version__ = "2.0.0"
__author__ = "HierHGT-DTI Team"

__all__ = [
    "HierHGTDTIModel",
    "SharedNodeStem",
    "AtomEncoder",
    "ResidueEncoder",
    "PackedHGTConv",
    "PackedJointHGT",
    "is_torch_scatter_available",
    "HierHGTDTIDataset",
    "hierhgt_dti_collate_fn",
]
