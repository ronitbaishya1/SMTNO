# GitHub setup (existing repository)

The live repository is `https://github.com/ronitbaishya1/SMTNO`. Do **not** initialize another Git repository inside the existing project.

To clone into your current Mac research folder without overwriting FEM data:

```bash
cd "/Users/ronitbaishya/Desktop/IIT Guwahati/SMTNO"
git clone https://github.com/ronitbaishya1/SMTNO.git github_source
```

If `github_source/.git` already exists, update with:

```bash
git -C github_source pull --ff-only
```

Your working FreeFEM++ sources are under `FEM/`; the GitHub copy is organized under `src/`.
Use `git diff` or `rsync -avn` before copying code from one layout to the other.
Generated data and results remain outside GitHub.
