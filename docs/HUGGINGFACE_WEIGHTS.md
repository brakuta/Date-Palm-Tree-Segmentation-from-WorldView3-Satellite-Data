# Creating a Hugging Face account and uploading weights

Model weights are hosted on Hugging Face, not in the Git repository. This keeps
GitHub clean (no files over 100 MB) and gives users a standard, resumable
download via `palmseg download`.

---

## Part 1 — Create a Hugging Face account

1. Go to **https://huggingface.co** and click **Sign Up**.
2. Enter your email, choose a username, and create a password.
   - Your username becomes part of the model URL:
     `huggingface.co/brakuta/date-palm-wv3-models`
   - Use the same username as your GitHub account for consistency.
3. Verify your email when the confirmation message arrives.
4. Sign in.

---

## Part 2 — Create a model repository on Hugging Face

A model repository on Hugging Face is a storage space for weights and model
cards, separate from your code repository on GitHub.

1. Click your profile picture (top-right) → **New Model**.
2. Fill in:
   - **Owner:** your username
   - **Model name:** `date-palm-wv3-models`
   - **License:** `cc-by-4.0`
   - **Visibility:** Public
3. Click **Create model**.

The repository URL will be:
`https://huggingface.co/brakuta/date-palm-wv3-models`

---

## Part 3 — Prepare your weights locally

Your trained checkpoints come out of MMSegmentation as
`work_dirs/<config_name>/iter_XXXXXX.pth`. Before uploading, copy them to the
stable names the toolkit expects. Use the rename script:

```bash
# From the toolkit root, list the expected names for every model:
python scripts/rename_weights.py

# Then copy each checkpoint to its target name (one line per model):
python scripts/rename_weights.py --out weights \
    segformer_b5_ms=work_dirs/ALL-segformer_mit-b5.../iter_100000.pth \
    segformer_b5_rgb=work_dirs/segformer_mit-b5.../iter_100000.pth \
    upernet_swin_b_ms=work_dirs/All-upernet_swin_b.../iter_100000.pth \
    upernet_swin_b_rgb=work_dirs/upernet_swin_b.../iter_100000.pth \
    upernet_vit_deit_s_ms=work_dirs/All-vit_deit.../iter_95000.pth \
    upernet_vit_deit_s_rgb=work_dirs/vit_deit.../iter_90000.pth \
    uniformer_base_ms=work_dirs/All-uniformer_fpn_global_base.../iter_100000.pth \
    uniformer_base_rgb=work_dirs/uniformer_fpn_global_base.../iter_100000.pth \
    mask2former_swin_s_ms=work_dirs/All-mask2former_swin-s.../iter_100000.pth \
    mask2former_swin_s_rgb=work_dirs/mask2former_swin-s.../iter_100000.pth
```

After this, `weights/` should contain files like `segformer_b5_ms.pth`,
`upernet_swin_b_ms.pth`, and so on. Only copy models you actually trained and
validated — you do not need to upload every entry in the manifest.

Verify the folder:
```bash
ls -lh weights/
```

---

## Part 4 — Install the Hugging Face CLI

```bash
pip install -U huggingface_hub
```

---

## Part 5 — Log in to Hugging Face from the terminal

1. Go to **https://huggingface.co/settings/tokens** and click
   **New token**.
2. Give it a name (e.g. "upload from workstation"), set **Role** to **Write**,
   and click **Generate a token**.
3. Copy the token (it starts with `hf_...`).
4. In your terminal:

```bash
huggingface-cli login
```

Paste the token when prompted. You will see "Login successful".

---

## Part 6 — Upload the weights

```bash
# Upload the entire weights/ folder to the model repository.
# Replace brakuta with your actual Hugging Face username.
huggingface-cli upload brakuta/date-palm-wv3-models \
    weights/ . --repo-type model
```

This uploads every `.pth` file in `weights/` to the root of the model
repository. The files keep their local names, which must match the manifest.

If you have only some models ready, upload them individually:

```bash
huggingface-cli upload brakuta/date-palm-wv3-models \
    weights/segformer_b5_ms.pth segformer_b5_ms.pth --repo-type model

huggingface-cli upload brakuta/date-palm-wv3-models \
    weights/segformer_b5_rgb.pth segformer_b5_rgb.pth --repo-type model
```

Progress is shown for each file. Uploads are resumable — if interrupted, run
the command again and it will continue from where it stopped.

---

## Part 7 — Write a model card

A model card is a README that appears on the Hugging Face page and explains what
the models are, how to use them, and their accuracy.

1. On your Hugging Face model repository page, click **Edit model card**.
2. Replace the placeholder text with the following template (customise the
   accuracy table with your actual results):

```markdown
---
license: cc-by-4.0
language:
  - en
tags:
  - remote-sensing
  - semantic-segmentation
  - date-palm
  - worldview-3
  - mmsegmentation
---

# Date Palm Segmentation — WorldView-3 Models

Trained weights for the date palm semantic segmentation toolkit at
https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data

## Models

| ID | Architecture | Modality | mIoU | Mean F-score |
|----|-------------|---------|------|-------------|
| segformer_b5_ms | SegFormer MiT-B5 | 8-band MS | 77.8 | 86.0 |
| upernet_swin_b_ms | UPerNet Swin-B | 8-band MS | 78.1 | 86.2 |
| ... | | | | |

Fill in your numbers from the test set. See the repository for the
full model list including RGB variants.

## Usage

```bash
pip install -e git+https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data.git
palmseg download segformer_b5_ms
palmseg segment --model segformer_b5_ms --checkpoint weights/segformer_b5_ms.pth \
                --input scene.tif --out out/
```

## Reference

Al-Ruzouq et al. (2024), Ecological Indicators 163, 112110.
https://doi.org/10.1016/j.ecolind.2024.112110
```

3. Click **Save**.

---

## Part 8 — Link the toolkit to the Hugging Face repository

Edit `palmseg/weights_manifest.py` in your local toolkit folder:

```python
# Line 22 — replace with your actual username and repo name
HF_REPO_ID = 'brakuta/date-palm-wv3-models'
```

Commit and push to GitHub:

```bash
git add palmseg/weights_manifest.py
git commit -m "Set Hugging Face model repository"
git push
```

Users can now run:

```bash
palmseg download --list               # see available models
palmseg download segformer_b5_ms      # downloads to weights/
```

---

## Part 9 — Verify the download works

From a clean directory (without the local weights), test that the download
path works end to end:

```bash
mkdir /tmp/download_test && cd /tmp/download_test
palmseg download segformer_b5_ms --cache-dir .
palmseg inspect segformer_b5_ms.pth
# should print: in_channels: 8, num_classes: 2, modality: ms
```

---

## Part 10 — Keeping weights up to date

If you retrain a model and want to replace a checkpoint:

```bash
# Re-run the rename script for that model
python scripts/rename_weights.py --out weights \
    segformer_b5_ms=work_dirs/new_run/iter_100000.pth

# Re-upload the single file (overwrites the previous version)
huggingface-cli upload brakuta/date-palm-wv3-models \
    weights/segformer_b5_ms.pth segformer_b5_ms.pth --repo-type model
```

---

## Notes

- `.pth` files must not be committed to GitHub. They are already in `.gitignore`.
- Hugging Face keeps a version history; previous uploads are not permanently
  lost when you overwrite a file.
- For a citable, DOI-stamped archival copy, upload the weights to
  **Zenodo** (https://zenodo.org) as well and add the DOI to `CITATION.cff`.
  Keep Hugging Face for routine downloads.
