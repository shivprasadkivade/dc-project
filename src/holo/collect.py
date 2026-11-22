import random
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
from .config import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    RMS_THRESHOLD,
    DATA_DIR,
    BOUNDARY_INSIDE_DIR,
    BOUNDARY_OUTSIDE_DIR,
    CLASSES,
)
from .detector import ImpulseDetector, get_input_channels
from .audio_manager import SplitAudioEngine

def save_sample(save_dir: Path, label: str, waveform: np.ndarray) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(save_dir.glob("*.npy")))
    filename = save_dir / f"{label}_{existing:04d}.npy"
    np.save(filename, waveform)
    return filename

def collect_single_class(label: str, num_samples: int, threshold: float):
    save_dir = DATA_DIR / label
    detector = ImpulseDetector(threshold=threshold)
    print(f"\n[COLLECT - SINGLE CLASS] Recording {num_samples} samples for '{label}'...")
    if "left" in label or "right" in label:
        print(f"Target: Tap firmly in the CENTER zone of the {label.upper().replace('_', ' ')}.")
    else:
        print(f"Target: Produce ambient acoustic noise / keystrokes / clatter.")
    print("Tap whenever ready. Press Ctrl+C to stop.\n")

    count = 0
    try:
        with SplitAudioEngine(tap_callback=detector.audio_callback, channels=get_input_channels()) as engine:
            while count < num_samples:
                waveform = detector.event_queue.get()
                saved_file = save_sample(save_dir, label, waveform)
                count += 1
                peak = float(np.max(np.abs(waveform)))
                print(f"  Captured [{count}/{num_samples}] -> {saved_file.name} (Peak: {peak:.3f})")
    except KeyboardInterrupt:
        print(f"\n[Interrupted] Saved {count} samples for '{label}'.")
        return

    print(f"\nSaved {num_samples} center/gesture samples inside {save_dir}")

def collect_group_interleaved(samples_per_class: int, threshold: float):
    detector = ImpulseDetector(threshold=threshold)
    remaining = {c: samples_per_class for c in CLASSES}
    total_needed = samples_per_class * len(CLASSES)
    collected_total = 0

    print(f"\n{'='*65}")
    print(f" [COLLECT - RANDOMIZED CENTER GESTURE MODE]")
    print(f" Target: {samples_per_class} samples per class ({total_needed} total across {len(CLASSES)} classes)")
    print(f" Classes will cycle randomly in groups of 3 or 4 taps.")
    print(f" Press Ctrl+C at any time to exit safely.")
    print(f"{'='*65}\n")

    try:
        with SplitAudioEngine(tap_callback=detector.audio_callback, channels=get_input_channels()) as engine:
            while any(v > 0 for v in remaining.values()):
                active_classes = [c for c, rem in remaining.items() if rem > 0]
                current_class = random.choice(active_classes)
                group_size = min(random.choice([3, 4]), remaining[current_class])

                instructions = {
                    "left_tap": "Tap firmly in the CENTER of the LEFT palm rest",
                    "right_tap": "Tap firmly in the CENTER of the RIGHT palm rest",
                    "noise": "Produce ambient noise / key clicks / desk vibrations",
                }
                desc = instructions.get(current_class, f"Perform {current_class}")

                print(f"\n{'*'*65}")
                print(f">>> TARGET: [ {current_class.upper()} ] <<<")
                print(f"Action: {desc}")
                print(f"Group quota: {group_size} tap(s) | {remaining[current_class]} remaining for this class")
                print(f"{'*'*65}")

                for i in range(group_size):
                    waveform = detector.event_queue.get()
                    save_dir = DATA_DIR / current_class
                    saved_file = save_sample(save_dir, current_class, waveform)
                    remaining[current_class] -= 1
                    collected_total += 1
                    peak = float(np.max(np.abs(waveform)))
                    print(f"  [{i+1}/{group_size}] -> {saved_file.name} (Peak: {peak:.3f}) | Progress: {collected_total}/{total_needed}")

    except KeyboardInterrupt:
        print(f"\n\n[Collection Stopped by User]")
        print("Summary of samples recorded this session:")
        for c in CLASSES:
            done = samples_per_class - remaining[c]
            print(f"  - {c:10s}: {done}/{samples_per_class} captured")
        return

    print(f"\n{'='*65}")
    print(f"Successfully collected all {total_needed} samples across {len(CLASSES)} classes!")
    print(f"{'='*65}\n")

def collect_boundary_zone(zone: str, num_samples: int, threshold: float = RMS_THRESHOLD):
    """Collects boundary samples strictly into dataset/boundary/."""
    detector = ImpulseDetector(threshold=threshold)
    zone_clean = zone.lower().strip()
    if zone_clean in ["in", "inside"]:
        save_dir = BOUNDARY_INSIDE_DIR
        prefix = "boundary_inside"
        desc = "Inside boundary: Tap along the perimeter edges (outer edge, trackpad edge, top bezel) of left/right palm rests"
    elif zone_clean in ["out", "outside"]:
        save_dir = BOUNDARY_OUTSIDE_DIR
        prefix = "boundary_outside"
        desc = "Outside boundary: Knock/tap the desk surface around the laptop (2-15 cm) & upper chassis bezel"
    else:
        raise ValueError("Invalid boundary zone. Use 'inside' or 'outside'.")

    print(f"\n[COLLECT - BOUNDARY ZONE: {zone_clean.upper()}] Recording {num_samples} samples...")
    print(f"Action: {desc}")
    print(f"Target folder: {save_dir}")
    print("Waiting for strikes. Press Ctrl+C to stop.\n")

    count = 0
    try:
        with SplitAudioEngine(tap_callback=detector.audio_callback, channels=get_input_channels()) as engine:
            while count < num_samples:
                waveform = detector.event_queue.get()
                saved_file = save_sample(save_dir, prefix, waveform)
                count += 1
                peak = float(np.max(np.abs(waveform)))
                print(f"  [{count}/{num_samples}] Captured -> {saved_file.name} (Peak: {peak:.3f})")
    except KeyboardInterrupt:
        print(f"\n[Interrupted] Saved {count} boundary samples to {save_dir}.")
        return

    print(f"\nSaved {num_samples} boundary samples to {save_dir}\n")

def calibrate_geofence_session(threshold: float = RMS_THRESHOLD):
    """
    Rapid interactive calibration for Passive Acoustic Geofence.
    SAVES STRICTLY TO dataset/boundary/ (inside & outside).
    Center gesture classes (left_tap, right_tap) are NEVER mixed or overwritten.
    """
    detector = ImpulseDetector(threshold=threshold)
    phases = [
        (
            BOUNDARY_INSIDE_DIR,
            "boundary_left",
            15,
            "Phase 1/3: Tap firmly along the PERIMETER / BOUNDARY EDGES of your LEFT palm rest\n"
            "           (near trackpad border, outer chassis edge, top border). DO NOT tap the center.",
        ),
        (
            BOUNDARY_INSIDE_DIR,
            "boundary_right",
            15,
            "Phase 2/3: Tap firmly along the PERIMETER / BOUNDARY EDGES of your RIGHT palm rest\n"
            "           (near trackpad border, outer chassis edge, top border). DO NOT tap the center.",
        ),
        (
            BOUNDARY_OUTSIDE_DIR,
            "boundary_outside",
            15,
            "Phase 3/3: Knock/tap firmly on the DESK / TABLE surface around your laptop\n"
            "           (2-15 cm away on left, right, front) and on the upper chassis bezel above keys.",
        ),
    ]

    print("\n" + "=" * 70)
    print(" [ACOUSTIC GEOFENCE BOUNDARY CALIBRATION WIZARD]")
    print(" This wizard records spatial boundary samples to distinguish in-bounds")
    print(" palm rest taps from out-of-bounds desk bumps and external vibrations.")
    print(" All boundary data is saved separately in 'dataset/boundary/' without")
    print(" touching your center gesture datasets (dataset/left_tap, dataset/right_tap).")
    print("=" * 70)

    try:
        with SplitAudioEngine(tap_callback=detector.audio_callback, channels=get_input_channels()) as engine:
            for save_dir, prefix, count_target, instructions in phases:
                print(f"\n>>> {instructions}")
                print(f"Target: {count_target} taps | Saving to: {save_dir.relative_to(DATA_DIR.parent)} | Waiting for strikes...\n")
                count = 0
                while count < count_target:
                    waveform = detector.event_queue.get()
                    saved_file = save_sample(save_dir, prefix, waveform)
                    count += 1
                    peak = float(np.max(np.abs(waveform)))
                    print(f"  [{count}/{count_target}] Captured -> {saved_file.name} (Peak: {peak:.3f})")

        print("\n" + "=" * 70)
        print("Geofence boundary data collection complete!")
        print(f"  Inside boundary samples:  {len(list(BOUNDARY_INSIDE_DIR.glob('*.npy')))} files in {BOUNDARY_INSIDE_DIR.relative_to(DATA_DIR.parent)}")
        print(f"  Outside boundary samples: {len(list(BOUNDARY_OUTSIDE_DIR.glob('*.npy')))} files in {BOUNDARY_OUTSIDE_DIR.relative_to(DATA_DIR.parent)}")
        print("\nNext step: Run 'uv run python -m src.holo.train' to fit and serialize your model & geofence.")
        print("=" * 70 + "\n")
    except KeyboardInterrupt:
        print("\n[Interrupted] Geofence calibration aborted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect center training taps or calibrate passive acoustic geofence boundaries"
    )
    parser.add_argument(
        "--label", "--class",
        dest="label",
        type=str,
        default="all",
        help="Target gesture class label (left_tap, right_tap, noise, or 'all')"
    )
    parser.add_argument(
        "--samples", "--count",
        dest="samples",
        type=int,
        default=30,
        help="Number of taps to record (default: 30)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RMS_THRESHOLD,
        help="Mic RMS trigger threshold"
    )
    parser.add_argument(
        "--calibrate-geofence",
        action="store_true",
        help="Run 3-phase interactive calibration wizard for geofence boundaries (inside & outside)"
    )
    parser.add_argument(
        "--boundary",
        type=str,
        choices=["inside", "outside"],
        default=None,
        help="Collect boundary data specifically for 'inside' or 'outside' zone"
    )
    args = parser.parse_args()

    if args.calibrate_geofence:
        calibrate_geofence_session(args.threshold)
    elif args.boundary is not None:
        collect_boundary_zone(args.boundary, args.samples, args.threshold)
    elif args.label == "all":
        collect_group_interleaved(args.samples, args.threshold)
    else:
        collect_single_class(args.label, args.samples, args.threshold)
