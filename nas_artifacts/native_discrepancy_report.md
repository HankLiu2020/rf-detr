# RF-DETR-Seg-Small Native vs Paper

Runtime values are read from the constructed model/checkpoint; paper values are reference expectations only.

| Field | Paper expected | Runtime | Status |
|---|---:|---:|---|
| `resolution` | `384` | `384` | `PASS` |
| `patch_size` | `12` | `12` | `PASS` |
| `num_windows` | `2` | `2` | `PASS` |
| `decoder_layers` | `4` | `4` | `PASS` |
| `num_queries` | `100` | `100` | `PASS` |
| `backbone` | `DINOv2-S` | `dinov2_windowed_small` | `PASS` |

## Discrepancies

- None in paper-facing numeric fields.

Checkpoint metadata discrepancies are recorded in `native_architecture.json` and are not silently inferred.
