# Qwen2-VL-7B APP Evaluation Summary

Generated from the following logs (with rerun backfill):
- `logo_iid_qwen7b_1.log`
- `logo_iid_qwen7b_2.log`
- `logo_ood_qwen7b_1.log`
- `rerun_step.nohup.log`
- `rerun_failed_qwen7b.log`

Backfilled failed sub-runs:
- `ebay + entropy + step` -> `64.00%` (from `rerun_step.nohup.log`)
- `reminder + norm + step` -> `53.94%`
- `calendar + entropy + step` -> `54.22%`
- `google_maps + norm + episode` -> `5.00%`

## IID Results

| app | signal | step accuracy | episode accuracy |
|---|---|---:|---:|
| amazon | norm | 61.34% | 15.00% |
| amazon | entropy | 60.50% | 10.00% |
| clock | norm | 70.41% | 20.00% |
| clock | entropy | 73.47% | 25.00% |
| ebay | norm | 67.20% | 20.00% |
| ebay | entropy | 64.00% | 20.00% |
| etsy | norm | 62.71% | 20.00% |
| etsy | entropy | 63.56% | 20.00% |
| flipkart | norm | 66.67% | 20.00% |
| flipkart | entropy | 66.67% | 25.00% |
| google_drive | norm | 56.25% | 5.00% |
| google_drive | entropy | 56.25% | 0.00% |
| reminder | norm | 53.94% | 0.00% |
| reminder | entropy | 53.94% | 5.00% |
| youtube | norm | 54.72% | 15.00% |
| youtube | entropy | 58.49% | 15.00% |

## OOD Results

| app | signal | step accuracy | episode accuracy |
|---|---|---:|---:|
| adidas | norm | 51.85% | 0.00% |
| adidas | entropy | 52.59% | 0.00% |
| calendar | norm | 53.61% | 10.00% |
| calendar | entropy | 54.22% | 10.00% |
| decathlon | norm | 52.67% | 5.00% |
| decathlon | entropy | 51.15% | 0.00% |
| gmail | norm | 55.64% | 5.00% |
| gmail | entropy | 54.14% | 0.00% |
| google_maps | norm | 48.35% | 5.00% |
| google_maps | entropy | 47.80% | 5.00% |
| kitchen_stories | norm | 54.14% | 5.00% |
| kitchen_stories | entropy | 53.38% | 0.00% |

## Visualization

A visual table is generated at:
- `summary_qwen7b_app_visual.html`

Open with browser:
```bash
xdg-open /data0/piaohongming/jinyike/LoGO-internVL2/summary_qwen7b_app_visual.html
```
