# Per-lesion detection F1 for segmentation

Voxel overlap (Dice, IoU) answers "how much did the masks overlap?". For multi-focal disease, the clinical question is different: "**did it find each lesion?**" segauge answers that with per-lesion detection metrics: it labels connected components in the prediction and the ground truth, matches them one-to-one by IoU, and reports precision, recall, and F1.

```python
import segauge as sg

scores = sg.detection_scores(pred_mask, gt_mask, iou_threshold=0.1)
print(scores.precision, scores.recall, scores.f1)
print(scores.tp, scores.fp, scores.fn)   # detected, spurious, missed lesions
```

- A ground-truth lesion is **detected** (true positive) if a predicted component overlaps it with IoU ≥ `iou_threshold`.
- An unmatched prediction is a **false positive** (spurious finding).
- An unmatched ground-truth lesion is a **false negative** (missed lesion).

Matching is greedy by IoU and one-to-one, so two predictions cannot both claim the same lesion.

## Over a dataset, with confidence intervals

```python
import segauge as sg

result = sg.evaluate(cases, detection=True, detection_iou=0.1)
print(result.summary()["det_f1"])         # detection F1 with a CI
print(result.summary()["det_recall"])     # lesion-level sensitivity
```

See also: [Dice with confidence intervals](confidence-intervals.md), [the API reference](api.md).
