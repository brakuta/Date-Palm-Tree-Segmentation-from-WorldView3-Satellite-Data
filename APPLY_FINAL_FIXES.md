# Final-sweep patch (3 fixes + 1 regression test)

From the deep audit of the full codebase after the production push:

1. palmseg/tools/adapt_input_stem.py (+ its tests)
   torch.load now passes weights_only=False - on PyTorch >= 2.6 the new
   default refused to read mmengine checkpoints, breaking `palmseg adapt-stem`.
   (Same fix already applied earlier to the loader and converter; this call
   site was missed.)

2. palmseg/inference/tiled.py
   NaN nodata is now handled: float rasters that declare nodata as NaN failed
   the equality test, so NaN pixels were treated as valid and fed to the model
   (NaN logits -> garbage labels). NaN/Inf values are also zero-filled before
   inference so they cannot poison neighbouring pixels within a tile.

3. palmseg/pipeline.py
   The instance-raster profile now sets tiled=True with the block sizes,
   removing the GDAL CPLE_IllegalArg warning surfaced in end-to-end testing.

4. tests/test_production_hardening.py
   New regression test for the NaN-nodata read path (48 tests total).

## Apply (from the repository root)
    unzip -o final_fixes.zip -d .       # PowerShell: Expand-Archive final_fixes.zip -DestinationPath . -Force
    pytest -q                            # expect: 48 passed, 1 skipped
    ruff check .                         # expect: All checks passed!
    git add -A && git commit -m "Fix adapt-stem on torch>=2.6; handle NaN nodata in tiled inference; clean instance-raster profile" && git push
