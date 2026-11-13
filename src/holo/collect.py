import random
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
from .config import SAMPLE_RATE, BLOCK_SIZE, RMS_THRESHOLD, DATA_DIR, CLASSES
from .detector import ImpulseDetector, get_input_channels

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
    print("Tap firmly on the target zone whenever ready. Press Ctrl+C to stop.\n")

    count = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=detector.audio_callback
        ):
            while count < num_samples:
                waveform = detector.event_queue.get()
                saved_file = save_sample(save_dir, label, waveform)
                count += 1
                peak = float(np.max(np.abs(waveform)))
                print(f"  Captured [{count}/{num_samples}] -> {saved_file.name} (Peak: {peak:.3f})")
    except KeyboardInterrupt:
        print(f"\n[Interrupted] Saved {count} samples for '{label}'.")
        return

    print(f"\nSaved {num_samples} samples inside {save_dir}")

def collect_group_interleaved(samples_per_class: int, threshold: float):
    detector = ImpulseDetector(threshold=threshold)
    remaining = {c: samples_per_class for c in CLASSES}
    total_needed = samples_per_class * len(CLASSES)
    collected_total = 0

    print(f"\n{'='*65}")
    print(f" [COLLECT - RANDOMIZED GROUP MODE]")
    print(f" Target: {samples_per_class} samples per class ({total_needed} total across {len(CLASSES)} classes)")
    print(f" Classes will cycle randomly in groups of 3 or 4 taps.")
    print(f" Press Ctrl+C at any time to exit safely.")
    print(f"{'='*65}\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=detector.audio_callback
        ):
            while any(v > 0 for v in remaining.values()):
                active_classes = [c for c, rem in remaining.items() if rem > 0]
                current_class = random.choice(active_classes)
                group_size = min(random.choice([3, 4]), remaining[current_class])

                instructions = {
                    "left_tap": "Tap on the LEFT palm rest",
                    "right_tap": "Tap on the RIGHT palm rest",
                    "noise": "Produce ambient noise / key clicks / desk vibrations"
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

def calibrate_geofence_session(threshold: float = RMS_THRESHOLD):
    """
    Rapid 45-second interactive calibration for Generation 4 Passive Acoustic Geofence.
    Phase 1: 20 taps for left palm rest (center, inner edge, outer edge).
    Phase 2: 20 taps for right palm rest (center, inner edge, outer edge).
    Phase 3: 10 strikes for oob_desk (desk surface 2-15 cm around laptop & keyboard bezel).
    """
    detector = ImpulseDetector(threshold=threshold)
    phases = [
        ("left_tap", 20, "Phase 1/3: Tap across LEFT palm rest (center, inner edge, outer edge)"),
        ("right_tap", 20, "Phase 2/3: Tap across RIGHT palm rest (center, inner edge, outer edge)"),
        ("oob_desk", 10, "Phase 3/3: Knock/tap the DESK surface (2-15 cm around laptop) & upper keyboard bezel"),
    ]

    print("\n" + "=" * 70)
    print(" [RAPID 45-SECOND PASSIVE ACOUSTIC GEOFENCE CALIBRATION]")
    print(" This wizard records spatial boundary samples to establish your laptop's")
    print(" mechanical unibody manifold and eliminate desk-bump false triggers.")
    print("=" * 70)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=detector.audio_callback
        ):
            for label, count_target, instructions in phases:
                save_dir = DATA_DIR / label
                print(f"\n>>> {instructions}")
                print(f"Target: {count_target} taps | Waiting for strikes...\n")
                count = 0
                while count < count_target:
                    waveform = detector.event_queue.get()
                    saved_file = save_sample(save_dir, label, waveform)
                    count += 1
                    peak = float(np.max(np.abs(waveform)))
                    print(f"  [{count}/{count_target}] Captured -> {saved_file.name} (Peak: {peak:.3f})")

        print("\n" + "=" * 70)
        print("Geofence calibration data collection complete!")
        print("Run 'uv run python -m src.holo.train' to fit and serialize your geofence.")
        print("=" * 70 + "\n")
    except KeyboardInterrupt:
        print("\n[Interrupted] Geofence calibration aborted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect training taps or calibrate passive acoustic geofence boundaries"
    )
    parser.add_argument(
        "--label", "--class",
        dest="label",
        type=str,
        default="all",
        help="Target class label (e.g. left_tap, right_tap, noise, oob_desk, or 'all')"
    )
    parser.add_argument(
        "--samples", "--count",
        dest="samples",
        type=int,
        default=50,
        help="Number of taps to record per class (default: 50)"
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
        help="Run rapid 45-second interactive calibration wizard for geofence boundaries"
    )
    args = parser.parse_args()

    if args.calibrate_geofence:
        calibrate_geofence_session(args.threshold)
    elif args.label == "all":
        collect_group_interleaved(args.samples, args.threshold)
    else:
        collect_single_class(args.label, args.samples, args.threshold)
