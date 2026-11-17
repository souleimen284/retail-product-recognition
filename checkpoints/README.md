# Trained checkpoint

`best_model.pt` (about 6.3 MB) is **not stored in git**. It is published as a **GitHub Release asset**.

1. Download **[`best_model.pt`](https://github.com/souleimen284/retail-product-recognition/releases/download/v1.0/best_model.pt)** from the [v1.0 release](https://github.com/souleimen284/retail-product-recognition/releases/tag/v1.0), or in PowerShell:

   ```powershell
   Invoke-WebRequest -Uri https://github.com/souleimen284/retail-product-recognition/releases/download/v1.0/best_model.pt -OutFile checkpoints\best_model.pt
   ```

2. Place it here: `checkpoints/best_model.pt`.
3. Optional integrity check (PowerShell):

   ```powershell
   Get-FileHash checkpoints\best_model.pt -Algorithm SHA256
   ```

   Expected SHA-256 (also recorded in `results/metrics/metrics.json`):

   ```
   c64f9f98acf74ba7aa04b17c94c4d55a281e19a6fb513eb657ba192f738b70a3
   ```

Contents: MobileNetV3-Small weights (12-class head), the ordered class list
(index -> class name), preprocessing settings (224 px, ImageNet mean/std), training epoch (12) and
validation metrics. Load it with `src.model.load_checkpoint()` or use `python -m src.predict`.

Alternatively, retrain with `python -m src.train` (about 2-3 minutes on a 6-core CPU). A retrained
model can differ slightly on other hardware or library builds.
