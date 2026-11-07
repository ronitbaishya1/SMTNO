# Colab workflow

The original Google Drive project is under `My Drive/SMTNO/Colab Notebooks`. Source code is now stored on GitHub, while simulation datasets and checkpoints remain in private Drive folders.

Open `colab/SMTNO_Training_Colab.ipynb` in Google Colab. It clones:

`https://github.com/ronitbaishya1/SMTNO.git`

The notebook expects private datasets in `/content/drive/MyDrive/SMTNO/datasets/`. Change `DATA_ROOT` if your actual data location differs. Set CUDA runtime under **Runtime → Change runtime type**.

The GitHub `src/advection_diffusion/` folder currently contains source through Step 3C.1. Add the separately packaged Step 3C.2 Python files before running its MRMT training command from Colab. Do not upload datasets or tokens to GitHub.
