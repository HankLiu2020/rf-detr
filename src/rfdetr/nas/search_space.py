# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Generate and sample the dynamic native-bounded Small-Seg search space."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture


DEFAULT_RESOLUTIONS = (384, 448, 480, 512, 576, 640)
DEFAULT_PATCH_SIZES = (12, 16, 20)
DEFAULT_WINDOWS = (1, 2)
DEFAULT_QUERY_CANDIDATES = (50, 100, 200, 300)


@dataclass(frozen=True)
class SearchSpace:
    """All candidate records and their validity partitions."""

    candidates: tuple[ArchitectureSpec, ...]
    syntactically_valid: tuple[ArchitectureSpec, ...]
    invalid_divisibility: tuple[ArchitectureSpec, ...]
    decoder_zero_supported: bool

    def to_summary(self) -> dict[str, object]:
        return {
            "candidate_count": len(self.candidates),
            "syntactically_valid_count": len(self.syntactically_valid),
            "invalid_divisibility_count": len(self.invalid_divisibility),
            "decoder_zero_supported": self.decoder_zero_supported,
            "encoder_tuple_count": len({item.encoder_tuple for item in self.syntactically_valid}),
            "decoder_candidates": sorted({item.decoder_layers for item in self.syntactically_valid}),
            "query_candidates": sorted({item.num_queries for item in self.syntactically_valid}),
            "group_detr": sorted({item.group_detr for item in self.syntactically_valid}),
        }


def estimate_encoder_proposal_pool_size(
    resolution: int,
    patch_size: int,
    projector_scale: Iterable[str] = ("P4",),
) -> int:
    """Estimate the encoder proposal-pool width for the current model.

    RF-DETR's DINOv2 backbone emits a patch grid and ``MultiScaleProjector``
    changes its spatial size by a fixed P-level factor.  The native Small-Seg
    checkpoint uses only P4, whose factor is one.  Keeping this calculation
    explicit lets the controller perform an early guard while the transformer
    keeps the measured tensor width as the final runtime boundary.
    """

    factors = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
    grid = resolution // patch_size
    total = 0
    for level in projector_scale:
        scaled = int(grid * factors[level])
        total += scaled * scaled
    return total


def _candidate(
    native: NativeArchitecture,
    resolution: int,
    patch_size: int,
    num_windows: int,
    decoder_layers: int,
    num_queries: int,
    *,
    native_flag: bool,
    proposal_count: int,
) -> ArchitectureSpec:
    valid_geometry = resolution % (patch_size * num_windows) == 0
    reason = None if valid_geometry else "resolution_not_divisible_by_patch_size_times_num_windows"
    return ArchitectureSpec(
        resolution=resolution,
        patch_size=patch_size,
        num_windows=num_windows,
        decoder_layers=decoder_layers,
        num_queries=num_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
        native=native_flag,
        hardware_feasible=None,
        invalid_reason=reason,
        metadata={
            "native_num_select_policy": native.num_select,
            "postprocess_num_select_policy": "native_fixed",
            "encoder_proposal_pool_size": proposal_count,
            "encoder_proposal_pool_guard": proposal_count >= num_queries,
        },
    )


def generate_search_space(
    native: NativeArchitecture,
    *,
    resolutions: Iterable[int] = DEFAULT_RESOLUTIONS,
    patch_sizes: Iterable[int] = DEFAULT_PATCH_SIZES,
    num_windows: Iterable[int] = DEFAULT_WINDOWS,
    query_candidates: Iterable[int] = DEFAULT_QUERY_CANDIDATES,
    include_decoder_zero: bool = True,
    decoder_zero_supported: bool = True,
) -> SearchSpace:
    """Generate all proposed candidates plus the native tuple.

    Encoder candidates are filtered only by the official divisibility rule at
    this stage.  Query and decoder values are bounded by native capacity, while
    resolution and patch/window values remain elastic candidates.
    """

    resolutions_set = sorted({int(v) for v in resolutions} | {native.resolution})
    patches_set = sorted({int(v) for v in patch_sizes} | {native.patch_size})
    windows_set = sorted({int(v) for v in num_windows} | {native.num_windows})
    queries_set = sorted({int(v) for v in query_candidates if 0 < int(v) <= native.num_queries} | {native.num_queries})
    decoder_set = set(range(native.decoder_layers + 1))
    if not include_decoder_zero or not decoder_zero_supported:
        decoder_set.discard(0)

    all_specs: list[ArchitectureSpec] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for resolution in resolutions_set:
        for patch_size in patches_set:
            for window_count in windows_set:
                proposal_count = estimate_encoder_proposal_pool_size(
                    resolution,
                    patch_size,
                    native.projector_scale or ("P4",),
                )
                for decoder_layers in sorted(decoder_set):
                    for query_count in queries_set:
                        key = (resolution, patch_size, window_count, decoder_layers, query_count)
                        if key in seen:
                            continue
                        seen.add(key)
                        all_specs.append(
                            _candidate(
                                native,
                                resolution,
                                patch_size,
                                window_count,
                                decoder_layers,
                                query_count,
                                native_flag=(
                                    resolution == native.resolution
                                    and patch_size == native.patch_size
                                    and window_count == native.num_windows
                                    and decoder_layers == native.decoder_layers
                                    and query_count == native.num_queries
                                ),
                                proposal_count=proposal_count,
                            )
                        )

    all_specs.sort(key=lambda item: (item.encoder_tuple, item.decoder_layers, item.num_queries))
    syntactically_valid = tuple(item for item in all_specs if item.invalid_reason is None)
    invalid = tuple(item for item in all_specs if item.invalid_reason is not None)
    return SearchSpace(
        candidates=tuple(all_specs),
        syntactically_valid=syntactically_valid,
        invalid_divisibility=invalid,
        decoder_zero_supported=decoder_zero_supported and include_decoder_zero,
    )


def write_search_space(space: SearchSpace, output_dir: str | Path) -> None:
    """Write the required JSONL partitions and summary."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, values: Iterable[ArchitectureSpec]) -> None:
        path.write_text(
            "".join(json.dumps(value.to_dict(), sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )

    write_jsonl(root / "candidate_architectures.jsonl", space.candidates)
    write_jsonl(root / "syntactically_valid.jsonl", space.syntactically_valid)
    write_jsonl(root / "invalid_divisibility.jsonl", space.invalid_divisibility)
    (root / "search_space_summary.json").write_text(
        json.dumps(space.to_summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _group_by(values: Iterable[ArchitectureSpec], key: str) -> dict[object, list[ArchitectureSpec]]:
    groups: dict[object, list[ArchitectureSpec]] = {}
    for value in values:
        groups.setdefault(getattr(value, key), []).append(value)
    return groups


def sample_architecture(
    values: Iterable[ArchitectureSpec],
    *,
    seed: int,
    policy: str = "uniform_valid_arch",
) -> ArchitectureSpec:
    """Sample one architecture with deterministic uniform or balanced-patch policy."""

    candidates = list(values)
    if not candidates:
        raise ValueError("cannot sample an empty architecture set")
    generator = random.Random(seed)
    if policy == "uniform_valid_arch":
        return generator.choice(candidates)
    if policy != "balanced_patch":
        raise ValueError(f"unknown sampling policy {policy!r}")

    by_patch = _group_by(candidates, "patch_size")
    patch = generator.choice(sorted(by_patch))
    by_resolution = _group_by(by_patch[patch], "resolution")
    resolution = generator.choice(sorted(by_resolution))
    by_window = _group_by(by_resolution[resolution], "num_windows")
    window = generator.choice(sorted(by_window))
    return generator.choice(by_window[window])
