# Colab Commands for Four-Model Added Experiments

These commands copy the project from Drive/one storage into `/content`, run the split scripts, and copy only new result files back. LocateAnything is intentionally excluded for now because it needs a separate dependency environment.

## 1. Mount Drive and Copy Project from one/Drive to `/content`

```python
from google.colab import drive
drive.mount("/content/drive")
```

Set the source project path. If your one/Drive folder name is different, change only `ONE_PROJECT`.

```bash
ONE_PROJECT="/content/drive/MyDrive/vegetation_models_v2"
CONTENT_PROJECT="/content/vegetation_models_v2"

rm -rf "$CONTENT_PROJECT"
mkdir -p "$CONTENT_PROJECT"
rsync -a --info=progress2 "$ONE_PROJECT"/ "$CONTENT_PROJECT"/
cd "$CONTENT_PROJECT"
```

Check that DINO heads and scripts exist:

```bash
ls -lh 9_DINOv2/four_dataset_runs/*/best_head.pth
ls -lh 10_DINOv3/four_dataset_runs/*/best_head.pth
ls -lh scripts/colab_four_model_efficiency.py scripts/colab_four_model_perturbation.py
```

## 2. Install Common Dependencies

```bash
pip install -q -U transformers accelerate qwen-vl-utils thop opencv-python-headless
```

Do not install the LocateAnything pinned environment here. LocateAnything will be handled separately later.

## 3. Run Efficiency: DINOv2, DINOv3, QwenVL Only

Smoke test first:

```bash
cd /content/vegetation_models_v2
python scripts/colab_four_model_efficiency.py \
  --datasets zijinshan \
  --rounds-per-dataset 3 \
  --warmup 2 \
  --repeats 5 \
  --amp --channels-last --allow-tf32
```

Formal efficiency run:

```bash
cd /content/vegetation_models_v2
python scripts/colab_four_model_efficiency.py \
  --datasets zijinshan \
  --rounds-per-dataset 20 \
  --warmup 10 \
  --repeats 100 \
  --amp --channels-last --allow-tf32
```

Output:

```bash
results/four_model_efficiency_colab.csv
results/four_model_colab_path_report.json
```

## 4. Run Perturbation Robustness with the Same Settings as the Old Eight Models

The perturbation script uses exactly the old eight-model settings:

- brightness: `0.70`, `0.85`, `1.15`, `1.30`
- contrast: `0.75`, `0.90`, `1.10`, `1.25`
- gaussian_blur: `kernel_3`, `kernel_5`, `kernel_7`
- shadow_illumination: `mild`, `moderate`, `strong`

Smoke test for QwenVL with fewer prompt rounds:

```bash
cd /content/vegetation_models_v2
python scripts/colab_four_model_perturbation.py \
  --datasets zijinshan \
  --rounds-per-dataset 3 \
  --amp --channels-last --allow-tf32
```

Formal perturbation run:

```bash
cd /content/vegetation_models_v2
python scripts/colab_four_model_perturbation.py \
  --datasets zijinshan \
  --rounds-per-dataset 100 \
  --amp --channels-last --allow-tf32
```

If QwenVL is too slow or out of memory, run only DINOv2/DINOv3 first:

```bash
cd /content/vegetation_models_v2
python scripts/colab_four_model_perturbation.py \
  --datasets zijinshan \
  --skip-qwen \
  --amp --channels-last --allow-tf32
```

Then run QwenVL separately:

```bash
cd /content/vegetation_models_v2
python scripts/colab_four_model_perturbation.py \
  --datasets zijinshan \
  --skip-dino \
  --rounds-per-dataset 100 \
  --amp
```

Outputs:

```bash
results/four_model_robustness_colab.csv
results/four_model_prompt_detail_colab.csv
results/four_model_perturbed_prompt_images/
```

## 5. Copy New Results Back from `/content` to one/Drive

```bash
ONE_PROJECT="/content/drive/MyDrive/vegetation_models_v2"
CONTENT_PROJECT="/content/vegetation_models_v2"

mkdir -p "$ONE_PROJECT/results"
rsync -a "$CONTENT_PROJECT/results/four_model_efficiency_colab.csv" "$ONE_PROJECT/results/" || true
rsync -a "$CONTENT_PROJECT/results/four_model_robustness_colab.csv" "$ONE_PROJECT/results/" || true
rsync -a "$CONTENT_PROJECT/results/four_model_prompt_detail_colab.csv" "$ONE_PROJECT/results/" || true
rsync -a "$CONTENT_PROJECT/results/four_model_colab_path_report.json" "$ONE_PROJECT/results/" || true
rsync -a "$CONTENT_PROJECT/results/four_model_perturbed_prompt_images" "$ONE_PROJECT/results/" || true
```

Optional: copy the scripts back too, if they were edited in Colab.

```bash
rsync -a "$CONTENT_PROJECT/scripts/colab_four_model_efficiency.py" "$ONE_PROJECT/scripts/"
rsync -a "$CONTENT_PROJECT/scripts/colab_four_model_perturbation.py" "$ONE_PROJECT/scripts/"
```
