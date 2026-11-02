# AcousticTap V3 Hybrid Test Report (20261102_002127)

- **Winning Architecture:** `Hybrid_SVM_RBF` (70% ML + 30% Fourier Triangulation)
- **Best Tuned Parameters:** `{'clf__C': 5.0, 'clf__gamma': 'scale'}`
- **Dataset Partition:** 451 samples (360 train / 91 holdout)
- **Cross-Validation Macro F1:** `0.9640`
- **Holdout Test Accuracy:** `94.51%`
- **Holdout Test Macro F1:** `0.9450`

## Holdout Classification Report
```text
              precision    recall  f1-score   support

    left_tap       0.94      1.00      0.97        30
   right_tap       0.93      0.90      0.92        31
       noise       0.97      0.93      0.95        30

    accuracy                           0.95        91
   macro avg       0.95      0.95      0.94        91
weighted avg       0.95      0.95      0.94        91

```

## Confusion Matrix (`['left_tap', 'right_tap', 'noise']`)
```text
[[30  0  0]
 [ 2 28  1]
 [ 0  2 28]]
```
