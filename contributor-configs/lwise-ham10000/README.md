# L-WISE HAM10000

Dermoscopy classifier (L-WISE HAM10000 ResNet-50) with light Captum
explainers + Torchattacks robustness checks, tuned to run on CPU.

## Steps

1. **Get the model.** Download the L-WISE bundle:

   ```bash
   wget https://github.com/MorganBDT/L-WISE/releases/download/v1.0.0/L-Wise_ckpts.zip
   unzip L-Wise_ckpts.zip -d /tmp/lwise_ckpts
   ```

   Wrap it as an eager PyTorch module whose `forward` returns class logits
   (Grad-CAM needs forward hooks; TorchScript does not). Save to:

   ```
   contributor-configs/lwise-ham10000/artifacts/lwise_ham10000_eager.pt
   ```

   Override the path with `LWISE_HAM10000_MODEL=/abs/path/to/model.pt` if
   needed.

2. **Get the data.** Drop the balanced 21-image presentation subset and its
   labels under:

   ```
   contributor-configs/lwise-ham10000/artifacts/ham10000-presentation-224/
   contributor-configs/lwise-ham10000/artifacts/ham10000-presentation-labels.csv
   ```

3. **Run the demo.** From the repo root:

   ```bash
   contributor-configs/lwise-ham10000/run.sh           # POSIX
   contributor-configs\lwise-ham10000\run.ps1          # Windows
   ```

   The report lands at `<run-dir>/reports/lwise_ham10000_report.pdf`.

4. **Optional — MLflow tracking.** Use the `_mlflow` variant:

   ```bash
   uv run raitap \
     --config-dir contributor-configs/lwise-ham10000 \
     --config-name assessment_mlflow
   ```

   Local SQLite store at `mlflow/mlflow.db`; UI opens at
   `http://127.0.0.1:5001`. Forward to a remote MLflow server by overriding
   `tracking.output_forwarding_url=<url>`.

## What's in the report

- Grad-CAM (coarse lesion localisation)
- Saliency (fine-grained positive evidence)
- Occlusion (perturbation sanity check, coarse 56×56 grid)
- Low-step Integrated Gradients (smoother attribution baseline)
- FGSM + small-step PGD (bounded adversarial robustness)
- Metrics on the presentation subset — treat as sanity checks, not benchmarks.
