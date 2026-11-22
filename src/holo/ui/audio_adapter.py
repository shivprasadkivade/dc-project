"""Audio adapter for Holo UI.

Provides hardware device discovery and a robust SplitAudioEngine managing
the primary acoustic transient detector and secondary diagnostics audio stream.
Acts as a defensive adapter compatible with `src.holo.audio_manager`.
"""

from __future__ import annotations

import sys
import queue
import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from src.holo.config import SAMPLE_RATE, BLOCK_SIZE
from src.holo.detector import ImpulseDetector, get_input_channels

logger = logging.getLogger(__name__)


def list_audio_devices(force_rescan: bool = False) -> List[Dict[str, Any]]:
    """Enumerate actively connected audio input devices.

    Filters out disconnected Bluetooth profiles, kernel driver registry paths,
    virtual loopback mappers, and inactive endpoints.
    """
    devices: List[Dict[str, Any]] = []
    if sd is None:
        return [
            {
                "index": 0,
                "name": "Default System Microphone (Virtual)",
                "channels": 1,
                "samplerate": 44100.0,
                "is_default": True,
            }
        ]

    if force_rescan:
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            logger.debug("PortAudio re-initialization error: %s", e)

    seen_names = set()

    try:
        dev_list = sd.query_devices()
        default_in = None
        try:
            default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
        except Exception:
            pass

        for idx, dev in enumerate(dev_list):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in <= 0:
                continue

            raw_name = str(dev.get("name", ""))
            # 1. Clean whitespace and remove newlines/tabs
            clean_name = " ".join(raw_name.split())
            if not clean_name:
                continue

            # 2. Exclude Windows driver paths / registry artifacts (e.g. @System32, bthhfenum.sys)
            clean_lower = clean_name.lower()
            if "@system32" in clean_lower or ".sys" in clean_lower or "@" in clean_name:
                continue

            # 3. Exclude Windows virtual loopback mappers and speaker inputs
            if clean_name in ("Microsoft Sound Mapper - Input", "Primary Sound Capture Driver"):
                continue
            if clean_name.startswith("PC Speaker") or clean_name.startswith("Stereo Mix"):
                continue

            # 4. Verify that the device is actively connected by testing stream access
            sr = int(dev.get("default_samplerate", 44100))
            try:
                test_stream = sd.InputStream(
                    device=idx,
                    samplerate=sr,
                    channels=min(max_in, 1),
                    callback=lambda *a: None,
                )
                test_stream.start()
                test_stream.stop()
                test_stream.close()
            except Exception:
                # Device is disconnected (e.g. disconnected bluetooth, unplugged USB, or blocked)
                continue

            # 5. Deduplicate base device names across Host APIs (e.g. MME vs WASAPI)
            base_key = clean_lower.split("(")[0].strip() if "(" in clean_lower else clean_lower
            if base_key in seen_names and (idx != default_in):
                continue
            seen_names.add(base_key)

            is_def = (idx == default_in)
            devices.append({
                "index": idx,
                "name": clean_name,
                "channels": max_in,
                "samplerate": float(sr),
                "is_default": is_def,
            })

    except Exception as e:
        logger.warning("Failed to query audio devices: %s", e)

    if not devices:
        devices.append({
            "index": 0,
            "name": "Default Built-in Microphone",
            "channels": 1,
            "samplerate": 44100.0,
            "is_default": True,
        })

    return devices


def find_builtin_microphone(devices: Optional[List[Dict[str, Any]]] = None) -> Optional[int]:
    """Find the index of the primary built-in microphone."""
    devs = devices if devices is not None else list_audio_devices()
    if not devs:
        return None

    # Priority keywords for built-in mics on macOS / Windows / Linux
    builtin_keywords = ["macbook", "built-in", "builtin", "internal", "microphone array", "realtek"]
    for dev in devs:
        name_lower = dev["name"].lower()
        if any(kw in name_lower for kw in builtin_keywords):
            return int(dev["index"])

    # Fallback to the default input device
    for dev in devs:
        if dev.get("is_default"):
            return int(dev["index"])

    return int(devs[0]["index"])


def find_secondary_microphone(devices: Optional[List[Dict[str, Any]]] = None) -> Optional[int]:
    """Find the index of a secondary external microphone (USB, headset, etc.)."""
    devs = devices if devices is not None else list_audio_devices()
    if len(devs) <= 1:
        return None

    builtin_idx = find_builtin_microphone(devs)
    ext_keywords = ["usb", "headset", "external", "bluetooth", "airpods", "webcam"]

    for dev in devs:
        idx = int(dev["index"])
        if idx == builtin_idx:
            continue
        name_lower = dev["name"].lower()
        if any(kw in name_lower for kw in ext_keywords):
            return idx

    # If no keyword matched, return the first other input device
    for dev in devs:
        idx = int(dev["index"])
        if idx != builtin_idx:
            return idx

    return None


class SplitAudioEngine:
    """Manages audio capture for transient detection and continuous diagnostics."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        channels: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels if channels is not None else get_input_channels()

        self.detector = ImpulseDetector(threshold=threshold)
        self.secondary_queue: queue.Queue = queue.Queue(maxsize=200)

        self.tap_callback: Optional[Callable[[np.ndarray], None]] = None
        self.secondary_callback: Optional[Callable[[np.ndarray], None]] = None

        self.primary_device: Optional[int | str] = None
        self.secondary_device: Optional[int | str] = None

        self._stream: Optional[Any] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _audio_callback(self, indata, frames, time_info, status):
        """Unified audio callback servicing both transient detector and diagnostics queue."""
        block = indata.copy()

        # 1. Feed the primary transient impulse detector
        try:
            self.detector.audio_callback(indata, frames, time_info, status)
        except Exception as e:
            logger.error("Error in detector audio_callback: %s", e)

        # 2. Extract mono audio chunk for live visualization
        if block.ndim == 2 and block.shape[1] > 1:
            mono_chunk = np.mean(block, axis=1, dtype=np.float32)
        elif block.ndim == 2:
            mono_chunk = block[:, 0].astype(np.float32)
        else:
            mono_chunk = block.astype(np.float32)

        # Push to secondary queue with non-blocking overflow protection
        try:
            self.secondary_queue.put_nowait(mono_chunk)
        except queue.Full:
            try:
                self.secondary_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.secondary_queue.put_nowait(mono_chunk)
            except queue.Full:
                pass

        if self.secondary_callback is not None:
            try:
                self.secondary_callback(mono_chunk)
            except Exception as e:
                logger.error("Error in secondary_callback: %s", e)

    def start(
        self,
        primary_device: Optional[int | str] = None,
        secondary_device: Optional[int | str] = None,
    ) -> None:
        """Starts audio input stream."""
        if self._is_running:
            return

        self.primary_device = primary_device
        self.secondary_device = secondary_device

        if sd is None:
            logger.warning("sounddevice is not available. SplitAudioEngine running in synthetic mode.")
            self._is_running = True
            return

        try:
            dev_idx = None
            if primary_device is not None:
                try:
                    dev_idx = int(primary_device)
                except (ValueError, TypeError):
                    dev_idx = None

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                device=dev_idx,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
            logger.info("SplitAudioEngine started on device %s (SR=%d, Block=%d)", dev_idx, self.sample_rate, self.block_size)
        except Exception as e:
            self._is_running = False
            self._stream = None
            logger.error("Failed to start audio stream: %s", e)
            raise RuntimeError(f"Audio stream initialization failed: {e}") from e

    def stop(self) -> None:
        """Stops audio input stream gracefully."""
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None
        logger.info("SplitAudioEngine stopped.")


# Ensure src.holo.audio_manager resolves to this adapter if not present
if "src.holo.audio_manager" not in sys.modules:
    try:
        import src.holo.audio_manager  # type: ignore # noqa: F401
    except ImportError:
        sys.modules["src.holo.audio_manager"] = sys.modules[__name__]
