# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Optional sample-dynamics data structures.

The observer types in this module are deliberately independent from the model and optimizer.  They can therefore be
enabled for diagnostics without changing the default RF-DETR training path.
"""

from rfdetr.sample_dynamics.observation import PerSampleLossPacket, SampleObservationBuffer
from rfdetr.sample_dynamics.probe import (
    DeterministicProbeDataset,
    ProbeReport,
    ProbeSampleResult,
    match_predictions_to_target,
    run_deterministic_probe,
)

__all__ = [
    "DeterministicProbeDataset",
    "PerSampleLossPacket",
    "ProbeReport",
    "ProbeSampleResult",
    "SampleObservationBuffer",
    "match_predictions_to_target",
    "run_deterministic_probe",
]
