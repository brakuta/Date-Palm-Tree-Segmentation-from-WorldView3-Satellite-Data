# Released models

## Choosing a model: size tiers

Smaller variants use less GPU memory and run faster, which helps on limited
hardware, CPU-only inference, and large-area (country-scale) processing. For a
binary palm/background task at sub-metre resolution the accuracy gap to the
largest models is often small. Release and use whichever tier matches your
hardware and accuracy needs.

| Tier | Models | Use when |
|------|--------|----------|
| Light | `segformer_b0`, `upernet_swin_t` | limited VRAM, CPU, high-throughput |
| Standard | `segformer_b2`, `upernet_vit_deit_s` | balanced speed/accuracy |
| Best | `segformer_b5`, `upernet_swin_b`, `uniformer_base`, `mask2former_swin_s` | maximum accuracy, ample GPU |

Each is available in `_ms` (8-band) and `_rgb` form. The toolkit needs no code
change to add a variant: register it in `palmseg/weights_manifest.py` with its
config and checkpoint name. Report accuracy for any variant you release.


Each architecture is released in its highest variant, in both multispectral
("All", 8-band WorldView-3) and RGB form. Model ids match the manifest in
`palmseg/weights_manifest.py` and the config files in `configs/`.

| Model id                  | Architecture            | Modality | Config                              | Tree-counting* |
|---------------------------|-------------------------|----------|-------------------------------------|----------------|
| `segformer_b5_ms`         | SegFormer MiT-B5        | MS (8b)  | `configs/segformer_b5_ms.py`        | yes            |
| `upernet_swin_b_ms`       | UPerNet + Swin-Base     | MS (8b)  | `configs/upernet_swin_b_ms.py`      | yes            |
| `upernet_vit_deit_s_ms`   | UPerNet + ViT-DeiT-S16  | MS (8b)  | `configs/upernet_vit_deit_s_ms.py`  | yes            |
| `uniformer_base_ms`       | UniFormer-Base + FPN    | MS (8b)  | `configs/uniformer_base_ms.py`      | yes            |
| `mask2former_swin_s_ms`   | Mask2Former + Swin-S    | MS (8b)  | `configs/mask2former_swin_s_ms.py`  | mask only      |
| `segformer_b5_rgb`        | SegFormer MiT-B5        | RGB      | `configs/segformer_b5_rgb.py`       | yes            |
| `upernet_swin_b_rgb`      | UPerNet + Swin-Base     | RGB      | `configs/upernet_swin_b_rgb.py`     | yes            |
| `upernet_vit_deit_s_rgb`  | UPerNet + ViT-DeiT-S16  | RGB      | `configs/upernet_vit_deit_s_rgb.py` | yes            |
| `uniformer_base_rgb`      | UniFormer-Base + FPN    | RGB      | `configs/uniformer_base_rgb.py`     | yes            |
| `mask2former_swin_s_rgb`  | Mask2Former + Swin-S    | RGB      | `configs/mask2former_swin_s_rgb.py` | mask only      |

\* **Tree-counting** marks the conv-seg models whose per-class probability
surface drives the individual-tree heatmap step. Mask2Former uses a query-based
head; it is released for inference-to-mask reproduction. For individual-tree
counting and crown delineation, use a conv-seg model
(SegFormer / UPerNet-Swin / UPerNet-ViT / UniFormer).

## Reported accuracy (from the paper)

Multispectral integration improved segmentation over RGB. On the test set, the
multispectral models reported (mIoU / mean F-score):

- UniFormer: 77.88% / 86.01%
- UPerNet-Swin: 78.10% / 86.18%
- Mask2Former: 77.36% / 85.59%

On the held-out Dibba region (transferability), the multispectral models
reported mIoU of 84.36% (Mask2Former), 84.25% (UniFormer), and 83.17%
(UPerNet-Swin), with mean F-scores of 90.95% / 90.87% / 90.13%.

These are the published figures for context; reproduce them with `palmseg`
inference + the IoUMetric evaluator on your prepared test split.

## Architecture notes

- **num_classes / in_channels** in the config files are training defaults; the
  loader reconciles them to the checkpoint. A stale `num_classes=150` in any
  config is therefore harmless.
- **UniFormer** requires the custom `UniFormer` backbone module to be
  registered in your MMSegmentation install (it is not part of stock MMSeg).
  Vendor it into `mmseg/models/backbones/` or install the project's MMSeg
  fork at commit `b040e147`; the config references it by `type='UniFormer'`.
- **Input stem provenance:** the 8-band models were bootstrapped from
  ImageNet-pretrained backbones by expanding the 3-channel stem to 8 channels.
  See `palmseg/tools/adapt_input_stem.py` and `docs/FINETUNE.md`. This is not
  needed for inference — released checkpoints already contain the 8-channel
  stem.

## Weights hosting
Weights are distributed via Hugging Face Hub. Set `HF_REPO_ID` in
`palmseg/weights_manifest.py`, then:

```bash
palmseg download --all                 # or a single id
palmseg download segformer_b5_ms
```

Weights are licensed CC-BY-4.0; code is Apache-2.0.
