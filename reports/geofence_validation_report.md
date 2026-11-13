# Passive Acoustic Geofence Validation Report

Comprehensive evaluation of Generation 4 Two-Stage Passive Acoustic Geofencing.

---

## 1. Real-Time Verification Latency (1,000 Iterations)
Evaluated via high-precision `time.perf_counter_ns` over 1,000 consecutive calls to `AcousticGeofenceValidator.verify()`:

| Latency Metric | Measured Value | Standard Threshold | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | **0.0248 ms** | < 0.25 ms | **PASS (Sub-millisecond)** |
| **Median Latency (P50)** | **0.0242 ms** | < 0.25 ms | **PASS** |
| **95th Percentile (P95)** | **0.0288 ms** | < 0.40 ms | **PASS** |
| **99th Percentile (P99)** | **0.0373 ms** | < 0.50 ms | **PASS (Strict SLA Met)** |

The geofence spatial verification adds negligible computational overhead (0.0373 ms P99), keeping total algorithmic latency well within the 5.0 ms ceiling.

---

## 2. Boundary Separation & Out-of-Bounds Rejection Performance

| Boundary Metric | Target Standard | Measured Performance | Operational Significance |
| :--- | :---: | :---: | :--- |
| **In-Bounds True Acceptance Rate (TAR)** | > 95.0% | **100.0%** | Valid palm rest taps are preserved without user frustration. |
| **In-Bounds False Rejection Rate (FRR)** | < 5.0% | **0.0%** | Negligible false rejections on intentional taps. |
| **Out-of-Bounds Rejection Rate (OOB-RR)** | > 95.0% | **100.0%** | Desk bumps, table knocks, and bezel strikes are blocked. |
| **Active Mahalanobis Threshold ($\theta$)** | Config Scale | **5.59** | Calibrated Ledoit-Wolf boundary manifold. |

---

## 3. Rejection Breakdown by Gate Stage
- **Stage 1 (Physical Invariant Hard Gate):**
  - High-Frequency/Low-Frequency Energy Ratio (`stage1_hf`): 20 blocked
  - Attack-to-Decay Ratio (`stage1_adr`): 20 blocked
  - Onset Crest Factor (`stage1_crest`): 0 blocked
  - Lamb-Wave Dispersion Window (`stage1_disp`): 0 blocked
- **Stage 2 (Ledoit-Wolf Spatial Manifold Gate):**
  - Manifold Mahalanobis Outlier (`stage2_manifold`): 0 blocked

**Total Out-of-Bounds Blocked:** **40/40 (100.0%)**
