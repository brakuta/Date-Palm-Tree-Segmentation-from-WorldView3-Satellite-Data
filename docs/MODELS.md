# Released models

## Modality matters: use all eight bands

Across all five architectures, using the full eight WorldView-3 multispectral
bands outperformed RGB and RGB+NIR composites. In the source study (Al-Ruzouq et
al., 2024) the best multispectral models reached about 77-78% mIoU and 86% mean
F-score on the held-out test set, and 83-84% mIoU with 90-91% mean F-score on
independent Dibba-region scenes, each architecture gaining roughly 2% mIoU when
moving from RGB to eight bands. Prefer the `_ms` models when 8-band imagery is
available; the `_rgb` models exist for cases where only RGB is on hand.

Record the test-set mIoU and mean F-score for each of your released checkpoints
in the table below so users can choose on evidence.


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


## Released models

These are the weights published at `brakuta/date-palm-wv3-models` on Hugging Face,
in `.safetensors` format (tensor-only; the format the HF security scanner reports
as safe). `palmseg download <id>` fetches them by the weight-file name.

| Model id | Architecture | Modality | Head type | Weight file |
|----------|-------------|----------|-----------|-------------|
| `segformer_b3_ms` | SegFormer | MS | heatmap | `segformer_b3_ms.safetensors` |
| `segformer_b5_ms` | SegFormer | MS | heatmap | `segformer_b5_ms.safetensors` |
| `upernet_swin_s_ms` | UPerNet+Swin | MS | heatmap | `upernet_swin_s_ms.safetensors` |
| `upernet_swin_b_ms` | UPerNet+Swin | MS | heatmap | `upernet_swin_b_ms.safetensors` |
| `upernet_vit_deit_s_ms` | UPerNet+ViT-DeiT | MS | heatmap | `upernet_vit_deit_s_ms.safetensors` |
| `uniformer_fpn_global_ms` | UniFormer | MS | heatmap | `uniformer_fpn_global_ms.safetensors` |
| `uniformer_xs_ms` | UniFormer | MS | heatmap | `uniformer_xs_ms.safetensors` |
| `mask2former_swin_b_ms` | Mask2Former | MS | mask-only | `mask2former_swin_b_ms.safetensors` |
| `mask2former_swin_s_ms` | Mask2Former | MS | mask-only | `mask2former_swin_s_ms.safetensors` |
| `segformer_b3_rgb` | SegFormer | RGB | heatmap | `segformer_b3_rgb.safetensors` |
| `upernet_swin_t_rgb` | UPerNet+Swin | RGB | heatmap | `upernet_swin_t_rgb.safetensors` |
| `upernet_vit_deit_s_rgb` | UPerNet+ViT-DeiT | RGB | heatmap | `upernet_vit_deit_s_rgb.safetensors` |
| `uniformer_xs_rgb` | UniFormer | RGB | heatmap | `uniformer_xs_rgb.safetensors` |
| `mask2former_swin_t_rgb` | Mask2Former | RGB | mask-only | `mask2former_swin_t_rgb.safetensors` |

Download and inspect any model:

```bash
palmseg download segformer_b5_ms
palmseg inspect weights/segformer_b5_ms.safetensors   # channels / classes / modality
```

### UniFormer models need a custom backbone

`uniformer_fpn_global_ms`, `uniformer_xs_ms`, and `uniformer_xs_rgb` use the
`UniFormer` and `UniFormer_Light` backbones, which are not part of stock
MMSegmentation. Register these backbone modules in your MMSeg install before
using those models (the SegFormer, Swin, ViT-DeiT, and Mask2Former models need
nothing extra).

### Verify num_classes after download

The loader reads the class count from each checkpoint. If a model was trained
with a non-palm class count (for example a 150-class default left in a training
config), `palmseg inspect` will report it. Confirm `num_classes: 2` for the
binary palm task before running inference:

```bash
palmseg inspect weights/uniformer_fpn_global_ms.safetensors
```

