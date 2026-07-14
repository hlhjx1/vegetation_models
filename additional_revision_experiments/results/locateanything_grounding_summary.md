# LocateAnything Grounding Summary

This note records prompt-based inference/grounding results for later manuscript/revision use. QwenVL2.5-3B and LocateAnything were used only for inference: predicted boxes were converted into coarse binary masks for box-mask metrics. Neither model was trained or fine-tuned as a dense segmentation model.

## QwenVL2.5-3B Inference Box-Mask Results

| Dataset | Box-mask IoU | Box-mask F1 | Box-mask Acc |
|---|---:|---:|---:|
| Zijinshan | 0.6212 | 0.6981 | 0.7557 |
| LoveDA | 0.1175 | 0.1806 | 0.6437 |
| Potsdam | 0.3482 | 0.4233 | 0.6145 |
| Vaihingen | 0.4615 | 0.6080 | 0.5290 |

## LocateAnything Inference Box-Mask Results

Source files:

- `12_LocatingAnything/grounding_eval/locateanything_raw_outputs.jsonl`
- `results/locateanything_grounding_results.csv`

Each dataset contains 100 inference rounds, for 400 total runs.

| Dataset | n | Box-mask IoU | Box-mask F1 | Box-mask Acc | Avg. boxes |
|---|---:|---:|---:|---:|---:|
| LoveDA | 100 | 0.1122 | 0.1756 | 0.6649 | 16.89 |
| Potsdam | 100 | 0.5038 | 0.5914 | 0.6932 | 2.74 |
| Vaihingen | 100 | 0.5444 | 0.6924 | 0.7046 | 8.85 |
| Zijinshan | 100 | 0.5747 | 0.6457 | 0.7055 | 13.64 |
| ALL | 400 | 0.4338 | 0.5263 | 0.6920 | 10.53 |

## Notes

- The current pulled result file contains 400 CSV rows and 400 JSONL rows.
- Dataset counts are balanced: Zijinshan 100, LoveDA 100, Potsdam 100, Vaihingen 100.
- Fifteen runs returned `<box>None</box>`, which were counted as zero predicted boxes in the CSV metrics.
- Both QwenVL2.5-3B and LocateAnything results are supplemental prompt-based inference/grounding evidence and should not be mixed with the old eight dense segmentation model results.
