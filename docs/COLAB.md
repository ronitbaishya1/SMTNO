# Google Colab workflow

Use [`colab/SMTNO_Training_Colab.ipynb`](../colab/SMTNO_Training_Colab.ipynb) as a launcher for the version-controlled `.py` files. It clones `https://github.com/ronitbaishya1/SMTNO.git`, mounts your Google Drive, and launches training with CUDA when available.

The notebook expects datasets under `/content/drive/MyDrive/SMTNO/datasets/` and writes checkpoints, metrics and plots to `/content/drive/MyDrive/SMTNO/GPU_runs/`. Change `DATA_ROOT` and `OUTPUT_ROOT` inside the notebook if your Drive layout differs.

- Step 3C.1 uses the private `step3B4` dataset.
- Step 3C.2A repeats advection–diffusion training with three random seeds.
- Step 3C.2B uses the private `step3C2B` mobile–immobile transport dataset. Generate and audit it before training.

The older `colab/Project_Execution_legacy.ipynb` is preserved for research history; its cells refer to the original `MyDrive/SMTNO/Colab Notebooks` paths.

Never commit generated datasets, logs, model checkpoints, access tokens or outputs.
