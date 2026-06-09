# Changelog

## 0.1.0

Initial release.

### Capabilities
- Three operating modes: `segment` (model to label/heatmap), `extract`
  (individual trees from existing rasters), and `segment-and-extract`.
- Multispectral (8-band) and RGB models for five architectures: SegFormer,
  UPerNet+Swin, UPerNet+ViT-DeiT, UniFormer, Mask2Former.
- Single-image and folder-level batch processing with resume, per-file fault
  isolation, and CSV summaries.
- Fine-tuning support for 2 or more classes.

### Model loading
- Input channels, class count, and the per-modality preprocessor are read from
  the checkpoint and applied to the config at load time.
- `inspect` reports a checkpoint's channels, classes, and modality.

### Tree extraction
- Seeds from the probability surface, or from the mask distance transform when
  no heatmap is supplied.
- Instance assignment by seeded watershed (default) or exact Euclidean
  nearest-seed (`voronoi`).
- Connected peak plateaus collapse to a single seed per crown.
- Georeferenced vectorisation with optional IoU overlap deduplication.

### Inference
- Tiled reading with edge-tapered seam blending.
- Output label is the argmax of the blended per-class probabilities.
- Nodata pixels are excluded; RGB tiles are reversed to BGR to match training.
- Accumulator memory budget with a clear error on oversized inputs.

### Installation and tooling
- Raster I/O uses rasterio (bundled GDAL) with an osgeo.gdal fallback.
- `doctor` checks the raster backend, PyTorch/CUDA, the MM stack, and the
  package, and prints fixes.
- `adapt-stem` adapts a 3-channel backbone stem to N channels for fine-tuning.
- `prep-check` validates a prepared dataset.
- `tile_pipeline` builds MMSeg-format tiles from mosaics and polygon
  annotations.
- Custom dataset and transform register via config `custom_imports`; no files
  are copied into a cloned mmsegmentation.

### Tests
- pytest suite covering extraction, vectorisation, tiled-inference stitching,
  the stem adapter, configs, the demo, and diagnostics.
