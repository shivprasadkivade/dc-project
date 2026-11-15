from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from .config import CFG

@dataclass
class GeofenceDecision:
    """Result of passive acoustic geofence boundary verification."""
    passed: bool
    stage_failed: Optional[str]
    prob_inside: float
    dist_inside: float
    dist_outside: float
    threshold: float
    reason: str

def extract_subspace(ml_vec: np.ndarray, fourier_vec: np.ndarray) -> np.ndarray:
    """
    Extracts the scale-invariant 24-dimensional spatial geofencing subspace z in R^24
    sliced directly from the 70D ML feature vector and 43D Fourier vector.

    Subspace Dimensions:
    z[0]:     hf_lf_ratio (Fourier, in dB) - direct plate vs rubber-damped desk
    z[1]:     adr (ML Attack-to-Decay Ratio, in dB) - sharp metal rise vs soft desk bump
    z[2]:     crest_factor (ML Onset Crest Factor) - spiky metal transient vs rounded desk wave
    z[3]:     delta_t_disp (Fourier Lamb-wave dispersion lag, in ms)
    z[4:7]:   tau_low, tau_mid, tau_high (Fourier unwrapped phase group delays, 3D)
    z[7:11]:  Scale-invariant High-Frequency Onset Band Energies (Bands 8..11, centered, 4D)
    z[11:14]: First 3 Neighboring Sub-Band Ratios (Bands 0/1, 1/2, 2/3, 3D)
    z[14:16]: Mid-HF Onset-to-Wake Energy Ratio (Bands 8 and 9, 2D)
    z[16]:    Spectral Centroid mean (ML vector)
    z[17]:    Spectral Flatness mean (ML vector)
    z[18]:    Spectral Rolloff mean (ML vector)
    z[19]:    Transient SNR (ML vector)
    z[20]:    zcr_mean (ML Zero-Crossing Rate mean) - high for metal flexure, low for dull thumps
    z[21]:    lf_energy_share (Fourier relative low-frequency onset dominance: bands 0..2 vs 0..12)
    z[22]:    lag_samples (Fourier stereo GCC-PHAT time-difference-of-arrival)
    z[23]:    ild_db (Fourier stereo Interaural Level Difference in dB)
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

    # 6. Spectral Features (Centroid idx 60, Flatness idx 64, Rolloff idx 62, SNR idx 67)
    centroid_mean = ml[:, 60:61]
    flatness_mean = ml[:, 64:65]
    rolloff_mean = ml[:, 62:63]
    snr_db = ml[:, 67:68]

    # 7. Additional Physical & Spatial Discriminators (Indices 20..23)
    zcr_mean = ml[:, 65:66]
    lf_energy_share = np.mean(four[:, 0:3], axis=1, keepdims=True) - np.mean(four[:, 0:12], axis=1, keepdims=True)
    lag_samples = four[:, 41:42]
    ild_db = four[:, 42:43]

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
        rolloff_mean,
        snr_db,
        zcr_mean,
        lf_energy_share,
        lag_samples,
        ild_db,
    ])

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z if is_2d else z[0]

def _synthesize_outside_negatives(Z_in: np.ndarray, rng: np.random.Generator, n_synth: int = 100) -> np.ndarray:
    """
    Synthesizes physics-informed out-of-bounds desk knocks and external vibrations
    by cloning random in-bounds taps and applying rubber-foot low-pass attenuation,
    dispersion shifts, and cavity resonance damping.
    """
    n_samples = len(Z_in)
    if n_samples == 0:
        return np.empty((0, 24), dtype=np.float64)

    idx = rng.choice(n_samples, size=n_synth, replace=True)
    synth = Z_in[idx].copy()

    # Subtract 6.0..18.0 dB from z[:, 0] (hf_lf_ratio_db)
    synth[:, 0] -= rng.uniform(6.0, 18.0, size=n_synth)
    # Subtract 3.5..9.0 dB from z[:, 1] (adr)
    synth[:, 1] -= rng.uniform(3.5, 9.0, size=n_synth)
    # Multiply z[:, 2] (crest_factor) by 0.45..0.78
    synth[:, 2] *= rng.uniform(0.45, 0.78, size=n_synth)
    # Shift z[:, 3] (delta_t_disp) by [-1.0, 1.0] * 2.5..7.0 ms
    synth[:, 3] += rng.choice([-1.0, 1.0], size=n_synth) * rng.uniform(2.5, 7.0, size=n_synth)
    # Subtract 0.8..2.5 from z[:, 7:11] (hf_onset_centered)
    synth[:, 7:11] -= rng.uniform(0.8, 2.5, size=(n_synth, 4))
    # Multiply centroid_mean [16] and rolloff_mean [18] by 0.35..0.68
    synth[:, 16] *= rng.uniform(0.35, 0.68, size=n_synth)
    synth[:, 18] *= rng.uniform(0.35, 0.68, size=n_synth)
    # Multiply zcr_mean [20] by 0.35..0.65
    synth[:, 20] *= rng.uniform(0.35, 0.65, size=n_synth)
    # Add 0.6..1.8 to lf_energy_share [21]
    synth[:, 21] += rng.uniform(0.6, 1.8, size=n_synth)

    return np.nan_to_num(synth, nan=0.0, posinf=0.0, neginf=0.0)

class AcousticGeofenceValidator:
    """
    Generation 4 Two-Stage Passive Acoustic Geofence for laptop chassis tap spatial boundary verification.
    Stage 1: Fast physical invariant hard gate (HF/LF ratio, ADR, Crest factor, Lamb dispersion window, LF dominance).
    Stage 2: Non-linear Class-Conditional Spatial Manifold & RBF Boundary Ensemble (trained on inside boundary vs outside desk samples + synthetic negatives).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or CFG
        gf_cfg = cfg.get("geofence", {}) if isinstance(cfg, dict) else {}

        self.enabled: bool = bool(gf_cfg.get("enabled", True))
        self.boundary_threshold: float = float(gf_cfg.get("boundary_threshold", 0.62))
        self.max_inside_distance: float = float(gf_cfg.get("max_inside_distance", 5.80))
        self.surface_scale: float = float(gf_cfg.get("surface_tolerance_scale", 1.15))
        self.class_conditional: bool = bool(gf_cfg.get("class_conditional", True))

        # Default Stage 1 Thresholds
        self.cfg_min_hf: float = float(gf_cfg.get("min_hf_lf_ratio_db", -14.5))
        self.cfg_min_adr: float = float(gf_cfg.get("min_attack_decay_ratio", 3.50))
        self.cfg_min_crest: float = float(gf_cfg.get("min_crest_factor", 3.80))
        disp_win = gf_cfg.get("dispersion_window_ms", [-5.0, 3.0])
        self.cfg_disp_min: float = float(disp_win[0])
        self.cfg_disp_max: float = float(disp_win[1])

        # Active Stage 1 Thresholds (auto-calibrated during fit)
        self.min_hf_lf_ratio: float = self.cfg_min_hf
        self.min_adr: float = self.cfg_min_adr
        self.min_crest: float = self.cfg_min_crest
        self.disp_min: float = self.cfg_disp_min
        self.disp_max: float = self.cfg_disp_max
        self.max_lf_dominance: float = 2.50

        # Stage 2 Model State
        self.classifier: Optional[Pipeline] = None
        self.centroid_inside: Optional[np.ndarray] = None
        self.centroid_outside: Optional[np.ndarray] = None
        self.precision_inside: Optional[np.ndarray] = None
        self.zone_manifolds: List[Tuple[np.ndarray, np.ndarray]] = []
        self.threshold: float = self.max_inside_distance
        self.has_outside_data: bool = False
        self.is_fitted: bool = False

    def fit(
        self,
        inside_ml: np.ndarray,
        inside_fourier: np.ndarray,
        outside_ml: Optional[np.ndarray] = None,
        outside_fourier: Optional[np.ndarray] = None,
        inside_labels: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> AcousticGeofenceValidator:
        """
        Calibrates Stage 1 physical invariant boundaries and fits Stage 2
        boundary discriminator using inside palm rest boundary samples vs outside desk samples.
        """
        if not self.enabled:
            self.is_fitted = True
            return self

        if len(inside_ml) == 0:
            raise ValueError("Geofence training requires at least 1 in-bounds sample!")

        # Positional compatibility if labels array was passed as 3rd positional argument
        if inside_labels is None and outside_ml is not None and outside_fourier is None:
            arr = np.asarray(outside_ml)
            if arr.ndim == 1:
                inside_labels = arr
                outside_ml = None

        Z_in = extract_subspace(inside_ml, inside_fourier)

        # -----------------------------------------------------------------
        # Stage 1: Auto-Calibrate Physical Invariant Thresholds (Safe Percentiles)
        # -----------------------------------------------------------------
        p2_hf = float(np.percentile(Z_in[:, 0], 2.5))
        p2_adr = float(np.percentile(Z_in[:, 1], 2.5))
        p2_crest = float(np.percentile(Z_in[:, 2], 2.5))
        p1_disp = float(np.percentile(Z_in[:, 3], 1.0))
        p99_disp = float(np.percentile(Z_in[:, 3], 99.0))

        has_outside = (
            outside_ml is not None
            and outside_fourier is not None
            and len(outside_ml) >= 2
            and len(outside_fourier) >= 2
        )

        target_hf = p2_hf - 1.5
        target_adr = p2_adr - 1.0
        target_crest = p2_crest - 0.35

        if has_outside:
            Z_real_out = extract_subspace(outside_ml, outside_fourier)
            q75_hf = float(np.percentile(Z_real_out[:, 0], 75.0))
            q75_adr = float(np.percentile(Z_real_out[:, 1], 75.0))
            q75_crest = float(np.percentile(Z_real_out[:, 2], 75.0))

            if q75_hf < p2_hf - 0.5:
                target_hf = (p2_hf + q75_hf) / 2.0
            if q75_adr < p2_adr - 0.5:
                target_adr = (p2_adr + q75_adr) / 2.0
            if q75_crest < p2_crest - 0.2:
                target_crest = (p2_crest + q75_crest) / 2.0

        p4_hf = float(np.percentile(Z_in[:, 0], 4.0))
        p4_adr = float(np.percentile(Z_in[:, 1], 4.0))
        p4_crest = float(np.percentile(Z_in[:, 2], 4.0))

        # Enforce physical safety floors while keeping at least 96% of Z_in
        self.min_hf_lf_ratio = min(p4_hf, max(self.cfg_min_hf, target_hf) if p2_hf >= self.cfg_min_hf else target_hf)
        self.min_adr = min(p4_adr, max(self.cfg_min_adr, target_adr) if p2_adr >= self.cfg_min_adr else target_adr)
        self.min_crest = min(p4_crest, max(self.cfg_min_crest, target_crest) if p2_crest >= self.cfg_min_crest else target_crest)
        self.disp_min = min(float(np.percentile(Z_in[:, 3], 1.0)), max(self.cfg_disp_min, p1_disp - 0.5))
        self.disp_max = max(float(np.percentile(Z_in[:, 3], 99.0)), min(self.cfg_disp_max, p99_disp + 0.5))
        self.max_lf_dominance = float(np.percentile(Z_in[:, 21], 98.5)) + 0.35

        # -----------------------------------------------------------------
        # Stage 2: Fit Multi-Cluster / Class-Conditional Spatial Manifolds
        # -----------------------------------------------------------------
        self.centroid_inside = np.mean(Z_in, axis=0)

        lw_global = LedoitWolf(assume_centered=False)
        lw_global.fit(Z_in)
        cov_global = lw_global.covariance_.copy()
        cov_global[11:16, 11:16] *= self.surface_scale
        self.precision_inside = np.linalg.pinv(cov_global)

        self.zone_manifolds = []
        if self.class_conditional:
            if inside_labels is not None and len(inside_labels) == len(Z_in):
                labels_arr = np.asarray(inside_labels)
                for lbl in np.unique(labels_arr):
                    zone_mask = (labels_arr == lbl)
                    if np.sum(zone_mask) >= 5:
                        Z_zone = Z_in[zone_mask]
                        c_zone = np.mean(Z_zone, axis=0)
                        lw_z = LedoitWolf(assume_centered=False)
                        lw_z.fit(Z_zone)
                        cov_z = lw_z.covariance_.copy()
                        cov_z[11:16, 11:16] *= self.surface_scale
                        p_zone = np.linalg.pinv(cov_z)
                        self.zone_manifolds.append((c_zone, p_zone))
            elif len(Z_in) >= 20:
                try:
                    km = KMeans(n_clusters=2, random_state=42, n_init=5).fit(Z_in)
                    for cluster_id in range(2):
                        c_mask = (km.labels_ == cluster_id)
                        if np.sum(c_mask) >= 5:
                            Z_zone = Z_in[c_mask]
                            c_zone = np.mean(Z_zone, axis=0)
                            lw_z = LedoitWolf(assume_centered=False)
                            lw_z.fit(Z_zone)
                            cov_z = lw_z.covariance_.copy()
                            cov_z[11:16, 11:16] *= self.surface_scale
                            p_zone = np.linalg.pinv(cov_z)
                            self.zone_manifolds.append((c_zone, p_zone))
                except Exception:
                    pass

        if not self.zone_manifolds:
            self.zone_manifolds.append((self.centroid_inside, self.precision_inside))

        # Compute minimum distance to any zone manifold
        in_dists = []
        for i in range(len(Z_in)):
            zi = Z_in[i]
            dists = []
            for c_z, p_z in self.zone_manifolds:
                diff = zi - c_z
                d = float(np.sqrt(max(0.0, np.dot(np.dot(diff, p_z), diff))))
                dists.append(d)
            in_dists.append(min(dists))
        in_dists = np.array(in_dists)

        d_in_975 = float(np.percentile(in_dists, 97.5)) if len(in_dists) > 0 else 5.0
        self.threshold = min(self.max_inside_distance, d_in_975 * 1.10)

        # -----------------------------------------------------------------
        # Stage 2: Supervised Boundary Classifier + Synthetic Augmentation
        # -----------------------------------------------------------------
        rng = np.random.default_rng(42)
        Z_synth_out = _synthesize_outside_negatives(Z_in, rng=rng, n_synth=max(100, len(Z_in)))

        has_outside = (
            outside_ml is not None
            and outside_fourier is not None
            and len(outside_ml) >= 2
            and len(outside_fourier) >= 2
        )

        if has_outside:
            Z_real_out = extract_subspace(outside_ml, outside_fourier)
            # Weight real outside desk samples 3x via repetition
            Z_real_rep = np.vstack([Z_real_out] * 3)
            Z_out_all = np.vstack([Z_real_rep, Z_synth_out])
            self.has_outside_data = True
        else:
            Z_out_all = Z_synth_out
            self.has_outside_data = True

        self.centroid_outside = np.mean(Z_out_all, axis=0)

        X_bound = np.vstack([Z_in, Z_out_all])
        y_bound = np.hstack([np.ones(len(Z_in), dtype=int), np.zeros(len(Z_out_all), dtype=int)])

        # Train calibrated RBF boundary ensemble (SVC RBF + ExtraTrees + Logistic Regression)
        clf_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", VotingClassifier(
                estimators=[
                    ("svm", SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                    ("et", ExtraTreesClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=42)),
                    ("lr", LogisticRegression(C=0.8, class_weight="balanced", random_state=42, max_iter=500)),
                ],
                voting="soft",
                weights=[0.50, 0.35, 0.15],
            )),
        ])
        clf_pipe.fit(X_bound, y_bound)
        self.classifier = clf_pipe

        self.is_fitted = True
        return self

    def verify(
        self,
        ml_vec: np.ndarray,
        fourier_vec: np.ndarray,
        target_label: Optional[str] = None,
        *args,
        **kwargs,
    ) -> GeofenceDecision:
        """
        Verifies whether an acoustic transient originated inside the physical palm rest geofence.
        Executes in low-latency two-stage gating.
        """
        if not self.enabled or not self.is_fitted:
            return GeofenceDecision(
                passed=True,
                stage_failed=None,
                prob_inside=1.0,
                dist_inside=0.0,
                dist_outside=0.0,
                threshold=self.boundary_threshold,
                reason="geofence_disabled",
            )

        z = extract_subspace(ml_vec, fourier_vec)

        # -----------------------------------------------------------------
        # Stage 1: Physical Invariant Hard Gates
        # -----------------------------------------------------------------
        min_hf = getattr(self, "min_hf_lf_ratio", self.cfg_min_hf)
        min_adr = getattr(self, "min_adr", self.cfg_min_adr)
        min_crest = getattr(self, "min_crest", self.cfg_min_crest)
        disp_min = getattr(self, "disp_min", self.cfg_disp_min)
        disp_max = getattr(self, "disp_max", self.cfg_disp_max)
        max_lf = getattr(self, "max_lf_dominance", 2.50)

        if z[0] < min_hf:
            return GeofenceDecision(
                passed=False,
                stage_failed="stage1_hf",
                prob_inside=0.0,
                dist_inside=99.0,
                dist_outside=0.0,
                threshold=self.boundary_threshold,
                reason=f"HF/LF ratio too low ({z[0]:.1f} dB < {min_hf:.1f} dB) - rubber-damped desk vibration",
            )
        if z[1] < min_adr:
            return GeofenceDecision(
                passed=False,
                stage_failed="stage1_adr",
                prob_inside=0.0,
                dist_inside=99.0,
                dist_outside=0.0,
                threshold=self.boundary_threshold,
                reason=f"Attack-decay ratio too low ({z[1]:.2f} dB < {min_adr:.2f} dB) - sluggish desk rise time",
            )
        if z[2] < min_crest:
            return GeofenceDecision(
                passed=False,
                stage_failed="stage1_crest",
                prob_inside=0.0,
                dist_inside=99.0,
                dist_outside=0.0,
                threshold=self.boundary_threshold,
                reason=f"Crest factor too low ({z[2]:.2f} < {min_crest:.2f}) - rounded non-metal wave",
            )
        if not (disp_min <= z[3] <= disp_max):
            return GeofenceDecision(
                passed=False,
                stage_failed="stage1_disp",
                prob_inside=0.0,
                dist_inside=99.0,
                dist_outside=0.0,
                threshold=self.boundary_threshold,
                reason=f"Lamb dispersion lag outside plate window ({z[3]:.1f} ms not in [{disp_min:.1f}, {disp_max:.1f}])",
            )
        if z[21] > max_lf:
            return GeofenceDecision(
                passed=False,
                stage_failed="stage1_lf_thump",
                prob_inside=0.0,
                dist_inside=99.0,
                dist_outside=0.0,
                threshold=self.boundary_threshold,
                reason=f"Excessive low-frequency thump energy ({z[21]:.2f} > {max_lf:.2f})",
            )

        # -----------------------------------------------------------------
        # Stage 2: Supervised Boundary Discriminator & Spatial Manifold
        # -----------------------------------------------------------------
        zone_manifolds = getattr(self, "zone_manifolds", [])
        if zone_manifolds:
            dists = []
            for c_z, p_z in zone_manifolds:
                diff = z - c_z
                d = float(np.sqrt(max(0.0, np.dot(np.dot(diff, p_z), diff))))
                dists.append(d)
            d_in = min(dists)
        else:
            diff_in = z - self.centroid_inside
            d_in = float(np.sqrt(max(0.0, np.dot(np.dot(diff_in, self.precision_inside), diff_in))))

        diff_out = z - self.centroid_outside if self.centroid_outside is not None else np.zeros_like(z)
        d_out = float(np.linalg.norm(diff_out))

        if self.classifier is not None:
            prob_inside = float(self.classifier.predict_proba(z.reshape(1, -1))[0, 1])

            if prob_inside >= self.boundary_threshold and d_in <= self.threshold:
                return GeofenceDecision(
                    passed=True,
                    stage_failed=None,
                    prob_inside=prob_inside,
                    dist_inside=d_in,
                    dist_outside=d_out,
                    threshold=self.boundary_threshold,
                    reason=f"inside palm rest boundary (prob_in={prob_inside:.2f}, d_in={d_in:.2f}/{self.threshold:.2f})",
                )
            else:
                failed_stage = "outside_boundary" if prob_inside < self.boundary_threshold else "stage2_manifold"
                return GeofenceDecision(
                    passed=False,
                    stage_failed=failed_stage,
                    prob_inside=prob_inside,
                    dist_inside=d_in,
                    dist_outside=d_out,
                    threshold=self.boundary_threshold,
                    reason=f"outside boundary: desk knock / external vibration (prob_in={prob_inside:.2f} < {self.boundary_threshold:.2f} or d_in={d_in:.2f} > {self.threshold:.2f})",
                )
        else:
            # Fallback when classifier not fitted
            if d_in <= self.threshold:
                return GeofenceDecision(
                    passed=True,
                    stage_failed=None,
                    prob_inside=1.0,
                    dist_inside=d_in,
                    dist_outside=d_out,
                    threshold=self.threshold,
                    reason=f"inside boundary manifold (d_in={d_in:.2f} <= {self.threshold:.2f})",
                )
            else:
                return GeofenceDecision(
                    passed=False,
                    stage_failed="stage2_manifold",
                    prob_inside=0.0,
                    dist_inside=d_in,
                    dist_outside=d_out,
                    threshold=self.threshold,
                    reason=f"outside boundary manifold (d_in={d_in:.2f} > {self.threshold:.2f})",
                )
