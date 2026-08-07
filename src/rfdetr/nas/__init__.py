# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Native-bounded NAS utilities for RF-DETR segmentation models.

The package intentionally keeps the search space smaller than or equal to the
loaded model.  It is a preparation layer: callers must explicitly opt in to
any formal search runner.
"""

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.nas.native_inspector import inspect_native_architecture

__all__ = ["ArchitectureSpec", "NativeArchitecture", "inspect_native_architecture"]
