# Preparing and uploading model weights

Trained checkpoints come out of training as `work_dirs/<config>/iter_*.pth`.
They are renamed to stable names and uploaded to a Hugging Face model repo, from
which `palmseg download` fetches them.

## 1. Rename checkpoints to the manifest names

The filenames must match the `checkpoint` field in
`palmseg/weights_manifest.py`. List the expected names:

```bash
python scripts/rename_weights.py            # prints model_id -> filename
```

Copy each trained checkpoint to its target name:

```bash
python scripts/rename_weights.py --out weights \
    segformer_b5_ms=work_dirs/ALL-segformer_mit-b5.../iter_100000.pth \
    segformer_b5_rgb=work_dirs/segformer_mit-b5.../iter_100000.pth \
    upernet_swin_b_ms=work_dirs/All-upernet_swin_b.../iter_100000.pth
    # ...one per model you trained
```

You only upload the models you actually trained and validated. Smaller variants
(SegFormer-B0/B2, Swin-Tiny) are registered in the manifest but optional; omit
them if you did not train them.

## 2. Create a Hugging Face model repo and upload

```bash
pip install -U huggingface_hub
huggingface-cli login        # token from https://huggingface.co/settings/tokens

huggingface-cli repo create date-palm-wv3-models --type model
huggingface-cli upload brakuta/date-palm-wv3-models weights/ . --repo-type model
```

The upload keeps each local filename in the repo, so they match the manifest.

## 3. Point the toolkit at the repo

Edit `palmseg/weights_manifest.py`:

```python
HF_REPO_ID = 'brakuta/date-palm-wv3-models'
```

Commit and push. Users can now run:

```bash
palmseg download --list
palmseg download segformer_b5_ms
```

## 4. Report accuracy

In `docs/MODELS.md`, fill in the mIoU / mean F-score for each released model on
your test split. Release only variants you can report numbers for; an unvalidated
checkpoint in the table is worse than omitting it.

## Notes

- `.pth` files are git-ignored and must not be committed to GitHub (it rejects
  files over 100 MB). They live only on Hugging Face.
- For an archival, citable copy, also deposit the weights on Zenodo to obtain a
  DOI; keep Hugging Face for routine downloads.
