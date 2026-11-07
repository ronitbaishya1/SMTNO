# Mac project setup

Your existing project folder is:

```text
/Users/ronitbaishya/Desktop/IIT Guwahati/SMTNO
```

**Do not clone directly into the existing nonempty SMTNO project folder.** Clone into a subfolder, then copy only source files into the existing FreeFEM++ working directory if needed.

```bash
cd "/Users/ronitbaishya/Desktop/IIT Guwahati/SMTNO"
git clone https://github.com/ronitbaishya1/SMTNO.git github_source
mkdir -p FEM/advection_diffusion_2d
rsync -avn github_source/src/advection_diffusion/ FEM/advection_diffusion_2d/
```

Review the dry-run list. Then, only if you want the GitHub versions to replace matching local scripts:

```bash
rsync -av github_source/src/advection_diffusion/ FEM/advection_diffusion_2d/
```

To update the cloned repository later:

```bash
cd "/Users/ronitbaishya/Desktop/IIT Guwahati/SMTNO/github_source"
git pull --ff-only
```

Generated `results/` and large datasets remain local. FreeFEM++ must be separately installed; `pip install` does not provide its solver.
