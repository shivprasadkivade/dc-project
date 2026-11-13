from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.covariance import LedoitWolf
from .config import CFG

@dataclass
class GeofenceDecision:
    """Result of a two-stage passive acoustic geofence boundary verification."""
    passed: bool
    stage_failed: Optional[str]
    mahalanobis_left: float
    mahalanobis_right: float
    active_distance: float
    threshold: float

def extract_subspace(ml_vec: np.ndarray, fourier_vec: np.ndarray) -> np.ndarray:
    """
    Extracts the scale-invariant 18-dimensional spatial geofencing subspace z in R^18
    sliced directly from the 70D ML feature vector and 43D Fourier vector.

    Subspace Dimensions:
    z[0]:    hf_lf_ratio (Fourier, in dB)
    z[1]:    adr (ML Attack-to-Decay Ratio)
    z[2]:    crest_factor (ML Onset Crest Factor)
    z[3]:    delta_t_disp (Fourier Lamb-wave dispersion lag, in ms)
    z[4:7]:  tau_low, tau_mid, tau_high (Fourier unwrapped phase group delays, 3D)
    z[7:11]: Scale-invariant High-Frequency Onset Band Energies (Bands 8..11, centered, 4D)
    z[11:14]: First 3 Neighboring Sub-Band Ratios (Bands 0/1, 1/2, 2/3, 3D)
    z[14:16]: Mid-HF Onset-to-Wake Energy Ratio (Bands 8 and 9, 2D)
    z[16:18]: Spectral Centroid mean & Spectral Flatness mean (ML vector, 2D)
    """
    is_2d = (ml_vec.ndim == 2)
    ml = np.atleast_2d(ml_vec).astype(np.float64)
    four = np.atleast_2d(fourier_vec).astype(np.float64)

    # 1. Physical Invariants
    # hf_to_lf_ratio is index 36 in Fourier vector (log10 energy ratio -> 10.0 * log10 is dB)
    hf_lf_db = 10.0 * four[:, 36:37]
    adr = ml[:, 69:70]
    crest = ml[:, 68:69]
    disp_ms = four[:, 35:36]

    # 2. Phase Group Delays (indices 37, 38, 39)
    group_delays = four[:, 37:40]

    # 3. High-Frequency Onset Bands 8..11 (indices 8..12), mean-centered across all 12 onset bands (0..12)
    onset_all = four[:, 0:12]
    onset_mean = np.mean(onset_all, axis=1, keepdims=True)
    hf_onset_centered = four[:, 8:12] - onset_mean

    # 4. Resonance Sub-band Ratios (indices 24..27: first 3 neighboring ratios)
    subband_ratios = four[:, 24:27]

    # 5. Mid-HF Onset-to-Wake Ratios for Bands 8 & 9 (onset: 8..10, wake: 20..22)
    onset_wake_ratio = four[:, 8:10] - four[:, 20:22]

    # 6. Spectral Centroid mean (idx 60) and Spectral Flatness mean (idx 64)
    centroid_mean = ml[:, 60:61]
    flatness_mean = ml[:, 64:65]

    z = np.hstack([
        hf_lf_db,
        adr,
        crest,
        disp_ms,
        group_delays,
        hf_onset_centered,
        subband_ratios,
        onset_wake_ratio,
        centroid_mean,
        flatness_mean,
    ])

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z if is_2d else z[0]

class AcousticGeofenceValidator:
    """
    Two-Stage Passive Acoustic Geofence for laptop chassis tap spatial verification.
    Stage 1: Fast physical invariant gate (HF/LF ratio, ADR, Crest factor, Lamb dispersion window).
    Stage 2: Ledoit-Wolf regularized spatial manifold gate in R^18 using precomputed precision matrices.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or CFG
        gf_cfg = cfg.get("geofence", {}) if isinstance(cfg, dict) else {}

        self.enabled: bool = bool(gf_cfg.get("enabled", True))
        self.class_conditional: bool = bool(gf_cfg.get("class_conditional", True))
        self.surface_scale: float = float(gf_cfg.get("surface_tolerance_scale", 1.15))

        # Default Stage 1 Thresholds
        self.cfg_min_hf: float = float(gf_cfg.get("min_hf_lf_ratio_db", -14.0))
        self.cfg_min_adr: float = float(gf_cfg.get("min_attack_decay_ratio", 1.85))
        self.cfg_min_crest: float = float(gf_cfg.get("min_crest_factor", 4.2))
        disp_win = gf_cfg.get("dispersion_window_ms", [-5.5, 2.5])
        self.cfg_disp_min: float = float(disp_win[0])
        self.cfg_disp_max: float = float(disp_win[1])

        # Active Stage 1 Thresholds (auto-calibrated during fit)
        self.min_hf_lf_ratio: float = self.cfg_min_hf
        self.min_adr: float = self.cfg_min_adr
        self.min_crest: float = self.cfg_min_crest
        self.disp_min: float = self.cfg_disp_min
        self.disp_max: float = self.cfg_disp_max

        # Stage 2 Manifold & Threshold
        self.cfg_threshold: float = float(gf_cfg.get("mahalanobis_threshold", 5.90))
        self.threshold: float = self.cfg_threshold

        self.centroids: Dict[str, np.ndarray] = {}
        self.precisions: Dict[str, np.ndarray] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        ml_features: np.ndarray,
        fourier_features: np.ndarray,
        labels: np.ndarray,
        oob_ml: Optional[np.ndarray] = None,
        oob_fourier: Optional[np.ndarray] = None,
    ) -> AcousticGeofenceValidator:
        """
        Calibrates Stage 1 physical invariant boundaries and fits Stage 2 Ledoit-Wolf
        spatial covariance manifolds for in-bounds palm rest classes.
        """
        if not self.enabled:
            self.is_fitted = True
            return self

        labels_arr = np.asarray(labels).astype(str)
        # Filter for in-bounds macro classes (exclude oob_* and noise)
        in_bounds_mask = np.array([
            not (lbl.startswith("oob_") or lbl == "noise") for lbl in labels_arr
        ])

        if not np.any(in_bounds_mask):
            # Fallback if all classes lack prefix
            in_bounds_mask = np.ones(len(labels_arr), dtype=bool)

        X_ml_in = ml_features[in_bounds_mask]
        X_four_in = fourier_features[in_bounds_mask]
        y_in = labels_arr[in_bounds_mask]

        Z_in = extract_subspace(X_ml_in, X_four_in)

        # -----------------------------------------------------------------
        # Stage 1: Auto-Calibrate Physical Invariant Thresholds
        # -----------------------------------------------------------------
        if len(Z_in) > 0:
            p1_hf = float(np.percentile(Z_in[:, 0], 1.0))
            p1_adr = float(np.percentile(Z_in[:, 1], 1.0))
            p1_crest = float(np.percentile(Z_in[:, 2], 1.0))
            p05_disp = float(np.percentile(Z_in[:, 3], 0.5))
            p995_disp = float(np.percentile(Z_in[:, 3], 99.5))

            # Set active floors to the more permissive of config and empirical p1 minus margin
            self.min_hf_lf_ratio = min(self.cfg_min_hf, p1_hf - 0.15 * max(1.0, abs(p1_hf)))
            self.min_adr = min(self.cfg_min_adr, p1_adr - 0.15 * max(1.0, abs(p1_adr)))
            self.min_crest = min(self.cfg_min_crest, p1_crest - 0.15 * max(1.0, abs(p1_crest)))
            self.disp_min = min(self.cfg_disp_min, p05_disp - 0.5)
            self.disp_max = max(self.cfg_disp_max, p995_disp + 0.5)

        # -----------------------------------------------------------------
        # Stage 2: Fit Ledoit-Wolf Spatial Covariance Manifolds
        # -----------------------------------------------------------------
        self.centroids.clear()
        self.precisions.clear()
        unique_classes = np.unique(y_in)

        in_distances = []
        for c in unique_classes:
            mask_c = (y_in == c)
            Z_c = Z_in[mask_c]
            if len(Z_c) < 2:
                continue

            lw = LedoitWolf(assume_centered=False)
            lw.fit(Z_c)
            cov = lw.covariance_.copy()

            # Apply surface boundary tolerance scaling to wake/modal ratios (indices 11..16)
            cov[11:16, 11:16] *= self.surface_scale

            loc = lw.location_.copy()
            prec = np.linalg.pinv(cov)

            self.centroids[c] = loc
            self.precisions[c] = prec

            # Compute in-bounds distances for threshold calibration
            diffs = Z_c - loc
            # vectorized Mahalanobis distance: sqrt(sum((diff @ prec) * diff, axis=1))
            dists = np.sqrt(np.maximum(0.0, np.sum(np.dot(diffs, prec) * diffs, axis=1)))
            in_distances.extend(dists)

        # -----------------------------------------------------------------
        # Calibrate Mahalanobis Threshold theta_geofence
        # -----------------------------------------------------------------
        d_in_98 = float(np.percentile(in_distances, 98.0)) if len(in_distances) > 0 else self.cfg_threshold

        has_oob = (oob_ml is not None and oob_fourier is not None and len(oob_ml) > 0 and len(oob_fourier) > 0)
        if has_oob and len(self.centroids) > 0:
            Z_oob = extract_subspace(oob_ml, oob_fourier)
            oob_dists = []
            for z_sample in Z_oob:
                sample_dists = []
                for c in self.centroids:
                    diff = z_sample - self.centroids[c]
                    prec = self.precisions[c]
                    sample_dists.append(float(np.sqrt(np.maximum(0.0, np.dot(np.dot(diff, prec), diff)))))
                oob_dists.append(min(sample_dists))

            d_oob_10 = float(np.percentile(oob_dists, 10.0))
            if d_oob_10 > d_in_98:
                self.threshold = float(np.sqrt(d_in_98 * d_oob_10))
            else:
                self.threshold = max(d_in_98 * 1.15, self.cfg_threshold)
        else:
            self.threshold = max(d_in_98 * 1.15, self.cfg_threshold)

        self.is_fitted = (len(self.centroids) > 0)
        return self

    def verify(
        self,
        ml_vec: np.ndarray,
        fourier_vec: np.ndarray,
        predicted_class: Optional[str] = None,
    ) -> GeofenceDecision:
        """
        Verifies whether an acoustic transient originated inside the physical palm rest geofence.
        Executes in < 0.50 ms using precomputed precision matrices.
        """
        if not self.enabled or not self.is_fitted:
            return GeofenceDecision(
                passed=True,
                stage_failed=None,
                mahalanobis_left=0.0,
                mahalanobis_right=0.0,
                active_distance=0.0,
                threshold=self.threshold,
            )

        z = extract_subspace(ml_vec, fourier_vec)

        # -----------------------------------------------------------------
        # Stage 1: Physical Invariant Hard Gates
        # -----------------------------------------------------------------
        if z[0] < self.min_hf_lf_ratio:
            return GeofenceDecision(False, "stage1_hf", 0.0, 0.0, 0.0, self.threshold)
        if z[1] < self.min_adr:
            return GeofenceDecision(False, "stage1_adr", 0.0, 0.0, 0.0, self.threshold)
        if z[2] < self.min_crest:
            return GeofenceDecision(False, "stage1_crest", 0.0, 0.0, 0.0, self.threshold)
        if not (self.disp_min <= z[3] <= self.disp_max):
            return GeofenceDecision(False, "stage1_disp", 0.0, 0.0, 0.0, self.threshold)

        # -----------------------------------------------------------------
        # Stage 2: Regularized Spatial Manifold Gate
        # -----------------------------------------------------------------
        distances: Dict[str, float] = {}
        for c, loc in self.centroids.items():
            diff = z - loc
            prec = self.precisions[c]
            # Fast quadratic form: diff^T @ prec @ diff
            d2 = float(np.dot(np.dot(diff, prec), diff))
            distances[c] = float(np.sqrt(max(0.0, d2)))

        # Resolve left and right distances for logging
        d_left = 0.0
        d_right = 0.0
        for c, dist in distances.items():
            c_lower = c.lower()
            if "left" in c_lower:
                d_left = dist
            elif "right" in c_lower:
                d_right = dist

        if len(distances) == 1:
            only_dist = list(distances.values())[0]
            d_left = d_left or only_dist
            d_right = d_right or only_dist

        # Select active distance
        if self.class_conditional and predicted_class and predicted_class in distances:
            active_d = distances[predicted_class]
        elif distances:
            active_d = min(distances.values())
        else:
            active_d = 0.0

        if active_d > self.threshold:
            return GeofenceDecision(False, "stage2_manifold", d_left, d_right, active_d, self.threshold)

        return GeofenceDecision(True, None, d_left, d_right, active_d, self.threshold)
