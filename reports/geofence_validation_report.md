# Passive Acoustic Geofence Validation Report

Comprehensive evaluation of Generation 4 Two-Stage Passive Acoustic Geofencing.

---

## 1. Real-Time Verification Latency (1,000 Iterations)
Evaluated via high-precision `time.perf_counter_ns` over 1,000 consecutive calls to `AcousticGeofenceValidator.verify()`:

| Latency Metric | Measured Value | Standard Threshold | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | **4.9255 ms** | < 8.0 ms | **PASS (Sub-10ms Ensemble)** |
| **Median Latency (P50)** | **4.9294 ms** | < 8.0 ms | **PASS** |
| **95th Percentile (P95)** | **5.0366 ms** | < 10.0 ms | **PASS** |
| **99th Percentile (P99)** | **5.1710 ms** | < 12.0 ms | **PASS (Strict SLA Met)** |

The geofence spatial verification executes an advanced RBF boundary ensemble (5.1710 ms P99), keeping total algorithmic latency well within real-time budget.

---

## 2. Boundary Separation & Out-of-Bounds Rejection Performance

| Boundary Metric | Target Standard | Measured Performance | Operational Significance |
| :--- | :---: | :---: | :--- |
| **In-Bounds True Acceptance Rate (TAR)** | > 95.0% | **100.0%** | Valid palm rest taps are preserved without user frustration. |
| **In-Bounds False Rejection Rate (FRR)** | < 5.0% | **0.0%** | Negligible false rejections on intentional taps. |
| **Out-of-Bounds Rejection Rate (OOB-RR)** | > 90.0% | **95.0%** | Desk bumps, table knocks, and bezel strikes are blocked. |
| **Active Mahalanobis Threshold ($\theta$)** | Config Scale | **2.64** | Calibrated Ledoit-Wolf boundary manifold. |

---

## 3. Rejection Breakdown by Gate Stage
- **Stage 1 (Physical Invariant Hard Gate):**
  - High-Frequency/Low-Frequency Energy Ratio (`stage1_hf`): 0 blocked
  - Attack-to-Decay Ratio (`stage1_adr`): 3 blocked
  - Onset Crest Factor (`stage1_crest`): 0 blocked
  - Lamb-Wave Dispersion Window (`stage1_disp`): 0 blocked
  - Low-Frequency Thump Dominance (`stage1_lf_thump`): 0 blocked
- **Stage 2 (Supervised Boundary Classifier & Spatial Manifold Gate):**
  - Supervised Boundary Probability Outlier (`outside_boundary`): 35 blocked
  - Manifold Mahalanobis Outlier (`stage2_manifold`): 0 blocked

**Total Out-of-Bounds Blocked:** **38/40 (95.0%)**
