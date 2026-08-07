# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Generate the native-derived RF-DETR-Seg-Small search-space artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfdetr.nas.architecture import NativeArchitecture
from rfdetr.nas.search_space import generate_search_space, write_search_space


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", default=Path("nas_artifacts/native_architecture.json"), type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/search_space"), type=Path)
    parser.add_argument("--decoder-zero-supported", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    native = NativeArchitecture(**json.loads(args.native.read_text(encoding="utf-8")))
    space = generate_search_space(native, decoder_zero_supported=args.decoder_zero_supported)
    write_search_space(space, args.output)
    print(json.dumps(space.to_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
