from __future__ import annotations

# Backward-compatible import shim. The original single-model implementation is now
# versioned as model1 so future model2/model3 variants can coexist cleanly.
from .model1 import (  # noqa: F401
    DEFAULT_SRU_DEPTH_WEIGHTS,
    Model1 as SruDualLidarAutoencoder,
    Model1Config as SruLidarAutoencoderConfig,
    Model1View as SruLidarViewAutoencoder,
)
