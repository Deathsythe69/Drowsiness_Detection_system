"""Evidence Video Recorder Module.

Captures pre-buffered and post-alert video footage when driver drowsiness is detected,
burns evidentiary telemetry watermarks (date/time, fatigue score, EAR/MAR, head pose, lighting),
and saves timestamped video clips to the 'evidence/' directory in background threads.
"""

import os
import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple
import cv2
import numpy as np


class EvidenceRecorder:
    """Thread-safe evidence video recorder with pre-alert rolling ring buffer."""

    def __init__(
        self,
        output_dir: str = "evidence",
        pre_buffer_seconds: float = 3.0,
        post_buffer_seconds: float = 5.0,
        fps: float = 20.0,
        cooldown_seconds: float = 15.0,
        codec: str = "mp4v",
        on_evidence_saved: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """Initialize evidence recorder.
        
        Args:
            output_dir: Directory to store evidence video files.
            pre_buffer_seconds: Seconds of video retained prior to alert trigger.
            post_buffer_seconds: Seconds of video recorded following alert trigger.
            fps: Video encoding frame rate.
            cooldown_seconds: Minimum delay between consecutive evidence recordings.
            codec: FourCC codec string (e.g. 'mp4v', 'XVID', 'MJPG').
            on_evidence_saved: Optional callback invoked when a video is written.
        """
        self.output_dir = output_dir
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_buffer_seconds = post_buffer_seconds
        self.fps = fps
        self.cooldown_seconds = cooldown_seconds
        self.codec = codec
        self.on_evidence_saved = on_evidence_saved

        # Ring buffer capacity
        self.buffer_capacity = max(30, int(self.pre_buffer_seconds * self.fps * 1.5))
        self._frame_buffer = deque(maxlen=self.buffer_capacity)
        self._lock = threading.Lock()

        # State tracking
        self._is_recording = False
        self._last_record_time = 0.0
        self._post_frames_needed = 0
        self._post_frames: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        self._current_trigger_metadata: Dict[str, Any] = {}

        # Ensure evidence folder exists
        os.makedirs(self.output_dir, exist_ok=True)

    def push_frame(self, frame: np.ndarray, telemetry: Optional[Dict[str, Any]] = None):
        """Add a frame and its associated telemetry metadata to the rolling buffer.
        
        Args:
            frame: OpenCV BGR image frame.
            telemetry: Dictionary containing runtime metrics (EAR, MAR, fatigue, etc.).
        """
        if frame is None or frame.size == 0:
            return

        telemetry = telemetry or {}
        telemetry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        with self._lock:
            # Always store in rolling pre-buffer
            self._frame_buffer.append((frame.copy(), dict(telemetry)))

            # If actively recording post-alert frames, collect them
            if self._is_recording:
                self._post_frames.append((frame.copy(), dict(telemetry)))
                if len(self._post_frames) >= self._post_frames_needed:
                    self._finalize_recording_async()

    def trigger_recording(
        self,
        trigger_reason: str = "drowsy_alert",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Trigger an evidence recording session if cooldown allows.
        
        Args:
            trigger_reason: Event identifier (e.g. 'drowsy_alert', 'microsleep').
            metadata: Supplemental details to record with the video.
            
        Returns:
            bool: True if recording was started, False if throttled by cooldown.
        """
        now = time.time()
        with self._lock:
            if self._is_recording:
                return False
            if now - self._last_record_time < self.cooldown_seconds:
                return False

            self._is_recording = True
            self._last_record_time = now
            self._post_frames_needed = max(10, int(self.post_buffer_seconds * self.fps))
            self._post_frames = []
            self._current_trigger_metadata = metadata or {}
            self._current_trigger_metadata["trigger_reason"] = trigger_reason
            self._current_trigger_metadata["trigger_time"] = datetime.now().isoformat()
            return True

    def _finalize_recording_async(self):
        """Compile pre-buffer and post-frames, and dispatch to background writer thread."""
        pre_frames = list(self._frame_buffer)
        post_frames = list(self._post_frames)
        metadata = dict(self._current_trigger_metadata)
        
        self._is_recording = False
        self._post_frames = []

        # Launch background writing thread
        thread = threading.Thread(
            target=self._write_video_worker,
            args=(pre_frames, post_frames, metadata),
            daemon=True
        )
        thread.start()

    def _apply_evidence_watermark(self, frame: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        """Render clear, tamper-evident telemetry banner and metadata overlay on frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Top Evidence Banner
        cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 24), cv2.FILLED)
        cv2.line(annotated, (0, 55), (w, 55), (0, 0, 220), 2)

        # Red recording dot & Title
        cv2.circle(annotated, (20, 28), 7, (0, 0, 255), -1)
        cv2.putText(
            annotated, "AI ATTENTION MONITOR - DROWSINESS EVIDENCE", (36, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Timestamp top right
        t_stamp = meta.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cv2.putText(
            annotated, t_stamp, (max(20, w - 240), 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA
        )

        # 2. Bottom Telemetry Overlay Bar
        cv2.rectangle(annotated, (0, h - 45), (w, h), (18, 18, 20), cv2.FILLED)
        cv2.line(annotated, (0, h - 45), (w, h - 45), (60, 60, 70), 1)

        fatigue = meta.get("fatigue_score", 0.0)
        ear = meta.get("ear", 0.0)
        mar = meta.get("mar", 0.0)
        pitch = meta.get("pitch", 0.0)
        motion = "DRIVING" if meta.get("is_vehicle_moving", True) else "STOPPED"
        light = "LOW LIGHT" if meta.get("is_low_light", False) else "NORMAL"

        status_str = (
            f"FATIGUE: {fatigue:.0f}/100 | EAR: {ear:.3f} | MAR: {mar:.2f} | "
            f"PITCH: {pitch:.1f}deg | VEHICLE: {motion} | LIGHT: {light}"
        )
        cv2.putText(
            annotated, status_str, (15, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA
        )

        # 3. Critical State Banner
        state_label = meta.get("state", "DROWSY_ALERT")
        if "ALERT" in state_label:
            cv2.rectangle(annotated, (w - 180, 65), (w - 10, 95), (0, 0, 200), cv2.FILLED)
            cv2.putText(
                annotated, "! DROWSY ALERT !", (w - 170, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA
            )

        return annotated

    def _write_video_worker(
        self,
        pre_frames: List[Tuple[np.ndarray, Dict[str, Any]]],
        post_frames: List[Tuple[np.ndarray, Dict[str, Any]]],
        metadata: Dict[str, Any]
    ):
        """Worker function that creates the video file on disk."""
        all_frames = pre_frames + post_frames
        if not all_frames:
            return

        sample_img = all_frames[0][0]
        h, w = sample_img.shape[:2]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evidence_{timestamp_str}_drowsy.mp4"
        filepath = os.path.join(self.output_dir, filename)

        # Fallback codecs
        fourcc_options = [
            ("mp4v", filepath),
            ("XVID", os.path.join(self.output_dir, f"evidence_{timestamp_str}_drowsy.avi")),
            ("MJPG", os.path.join(self.output_dir, f"evidence_{timestamp_str}_drowsy.avi"))
        ]

        writer = None
        saved_path = filepath

        for codec_str, path_opt in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                out = cv2.VideoWriter(path_opt, fourcc, self.fps, (w, h))
                if out.isOpened():
                    writer = out
                    saved_path = path_opt
                    break
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            print(f"Error: Could not open VideoWriter for evidence recording at {filepath}")
            return

        try:
            for frame, frame_meta in all_frames:
                watermarked = self._apply_evidence_watermark(frame, frame_meta)
                writer.write(watermarked)
        except Exception as e:
            print(f"Error while encoding evidence video: {e}")
        finally:
            writer.release()

        # Save companion JSON audit manifest
        json_path = os.path.splitext(saved_path)[0] + ".json"
        manifest_data = {
            "evidence_video": os.path.basename(saved_path),
            "created_at": datetime.now().isoformat(),
            "trigger_metadata": metadata,
            "total_frames": len(all_frames),
            "duration_seconds": len(all_frames) / max(1.0, self.fps),
            "resolution": [w, h]
        }
        try:
            with open(json_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
        except Exception:
            pass

        print(f"[EVIDENCE RECORDED] Successfully saved evidence clip to: {saved_path}")

        if self.on_evidence_saved is not None:
            try:
                self.on_evidence_saved(saved_path, manifest_data)
            except Exception as e:
                print(f"Error in on_evidence_saved callback: {e}")
