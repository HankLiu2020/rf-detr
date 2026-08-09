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
from rfdetr.sample_dynamics.review import ReviewExporter
from rfdetr.sample_dynamics.sampler import BucketQuotaSampler
from rfdetr.sample_dynamics.state import (
    SampleState,
    SampleStateRecord,
    SampleStateStore,
    StatePolicy,
    ema_update,
    percentile_rank,
    window_slope,
)
from rfdetr.sample_dynamics.weighting import SampleWeightPolicy, cap_effective_contribution

__all__ = [
    "DeterministicProbeDataset",
    "BucketQuotaSampler",
    "ema_update",
    "PerSampleLossPacket",
    "ProbeReport",
    "ProbeSampleResult",
    "ReviewExporter",
    "SampleObservationBuffer",
    "SampleState",
    "SampleStateRecord",
    "SampleStateStore",
    "SampleWeightPolicy",
    "cap_effective_contribution",
    "StatePolicy",
    "percentile_rank",
    "match_predictions_to_target",
    "run_deterministic_probe",
    "window_slope",
]
