# Model card — convnext_tiny

- Dataset fingerprint: `feb9fd7dce3cf1f6`
- Test images: 110
- Primary metric (average precision): 0.9325
- ROC-AUC: 0.9393
- Sensitivity / specificity at frozen threshold 0.4256: 0.9231 / 0.7586
- Threshold policy: `OOF development predictions; policy=youden; test labels never read`
- Input: EXIF-corrected RGB, fixed preprocessing and square multicrop inference.

## Limitations

- Small test cohort.
- Source/device/session metadata unavailable; those slices could not be assessed.
- No external OOD cohort is part of this split.
- Image-level performance is not clinical validation.
- Grad-CAM requires human A/B/C/D review and does not prove causal reasoning.
- This report is not clinical validation.
