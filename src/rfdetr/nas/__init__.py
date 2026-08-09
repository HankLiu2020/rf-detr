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
from rfdetr.nas.data_gate import TargetSplitManifestValidation, validate_target_split_manifest
from rfdetr.nas.evaluation import (
    SubnetEvaluation,
    SubnetEvaluator,
    benchmark_active_subnet_latency,
    evaluate_subnet,
    evaluate_subnet_pool,
)
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.nas.orchestration import (
    ExecutionLimits,
    FormalExecutionResult,
    build_formal_execution_plan,
    run_formal_execution_preparation,
)
from rfdetr.nas.pareto import compute_pareto_front, select_fast_balanced_accurate
from rfdetr.nas.ranking import (
    RankingGateResult,
    ShortFinetuneResult,
    run_proxy_search,
    run_ranking_gate,
    run_short_finetune_ranking,
    select_representative_architectures,
    spearman_rank_correlation,
)
from rfdetr.nas.training import SupernetTrainConfig, SupernetTrainResult, train_elastic_supernet

__all__ = [
    "ArchitectureSpec",
    "ExecutionLimits",
    "FormalExecutionResult",
    "NativeArchitecture",
    "RankingGateResult",
    "ShortFinetuneResult",
    "SubnetEvaluation",
    "SupernetTrainConfig",
    "SupernetTrainResult",
    "TargetSplitManifestValidation",
    "SubnetEvaluator",
    "benchmark_active_subnet_latency",
    "build_formal_execution_plan",
    "compute_pareto_front",
    "evaluate_subnet",
    "evaluate_subnet_pool",
    "inspect_native_architecture",
    "run_proxy_search",
    "run_formal_execution_preparation",
    "run_ranking_gate",
    "run_short_finetune_ranking",
    "select_fast_balanced_accurate",
    "select_representative_architectures",
    "spearman_rank_correlation",
    "train_elastic_supernet",
    "validate_target_split_manifest",
]
