# AcousticTap Hybrid Test Report (20261030_203146)

- **Winning Architecture:** `ExtraTrees_V3` (80% ML + 20% Fourier Triangulation)
- **Total Dataset Samples:** 1105 (884 train / 221 holdout)
- **5-Fold Cross-Validation Macro F1:** `0.8792`
- **Holdout Test Accuracy:** `87.33%`
- **Holdout Test Macro F1:** `0.8746`

## Holdout Classification Report
```text
              precision    recall  f1-score   support

    left_tap       0.85      0.87      0.86        91
       noise       0.81      0.96      0.88        45
   right_tap       0.95      0.84      0.89        85

    accuracy                           0.87       221
   macro avg       0.87      0.89      0.87       221
weighted avg       0.88      0.87      0.87       221

```

## Confusion Matrix (`['left_tap', 'noise', 'right_tap']`)
```text
[[79  9  3]
 [ 1 43  1]
 [13  1 71]]
```
