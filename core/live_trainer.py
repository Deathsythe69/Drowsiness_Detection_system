"""Live Active Learning and Continuous Trainer Module.

Provides continuous sample collection from running daytime and nighttime sessions,
aggregating real-world driver scenarios (open/closed eyes, yawns, head nods, low-light),
and enabling incremental self-training to continually improve model precision.
"""

import os
import time
import json
import random
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
import cv2
import numpy as np

from core.preprocessing import enhance_eye_region
from train_test.train_low_light_model import LowLightEyeCNN, preprocess_eye_sample, apply_night_augmentation


class LiveSampleCollector:
    """Thread-safe collector for live daytime and nighttime driver data."""

    def __init__(self, base_dir: str = "data/live_dataset"):
        self.base_dir = base_dir
        self._lock = threading.Lock()
        self.last_capture_times: Dict[str, float] = {}
        self.min_interval_seconds = 0.50  # Prevent duplicate frames
        self._ensure_directories()

    def _ensure_directories(self):
        """Create structured directory tree for daytime and nighttime scenarios."""
        categories = ["open", "closed", "yawn", "nod", "sunglasses", "regular_glasses", "general"]
        lighting_modes = ["day", "night"]
        for mode in lighting_modes:
            for cat in categories:
                path = os.path.join(self.base_dir, mode, cat)
                os.makedirs(path, exist_ok=True)

    def save_sample(
        self,
        eye_crop: np.ndarray,
        full_frame: Optional[np.ndarray],
        label: int,  # 0=closed/drowsy, 1=open/awake
        is_low_light: bool,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general",
        force: bool = False
    ) -> Optional[str]:
        """Save a labeled eye crop and metadata to the dataset directory.
        
        Args:
            eye_crop: Cropped eye image BGR or Gray.
            full_frame: Optional full context frame.
            label: 0 for closed/drowsy, 1 for open/awake.
            is_low_light: Whether frame was captured in low/no light.
            metadata: Telemetry dict (EAR, MAR, Pitch, Yaw, luminance, etc.).
            category: Subfolder category ('open', 'closed', 'yawn', 'nod', 'general').
            force: If True, bypasses rate limiting (e.g. manual user feedback).
            
        Returns:
            Optional[str]: Path to the saved image file or None if throttled.
        """
        if eye_crop is None or eye_crop.size == 0:
            return None

        current_time = time.time()
        key = f"{category}_{label}_{'night' if is_low_light else 'day'}"

        with self._lock:
            if not force:
                last_time = self.last_capture_times.get(key, 0.0)
                if (current_time - last_time) < self.min_interval_seconds:
                    return None
            self.last_capture_times[key] = current_time

        mode = "night" if is_low_light else "day"
        cat_dir = os.path.join(self.base_dir, mode, category)
        os.makedirs(cat_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_filename = f"live_{mode}_{category}_lbl{label}_{timestamp_str}"
        img_path = os.path.join(cat_dir, f"{base_filename}.png")
        json_path = os.path.join(cat_dir, f"{base_filename}.json")

        meta = {
            "timestamp": datetime.now().isoformat(),
            "label": int(label),
            "is_low_light": bool(is_low_light),
            "mode": mode,
            "category": category,
            "telemetry": metadata or {},
            "manual_tag": bool(force)
        }

        try:
            cv2.imwrite(img_path, eye_crop)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
            return img_path
        except Exception as e:
            print(f"Error saving live sample: {e}")
            return None

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Index dataset and return counts across day, night, open, and closed classes."""
        stats = {
            "total_samples": 0,
            "day_samples": 0,
            "night_samples": 0,
            "open_eyes": 0,
            "closed_eyes": 0,
            "manual_feedback_samples": 0,
            "categories": {}
        }

        if not os.path.exists(self.base_dir):
            return stats

        for root, _, files in os.walk(self.base_dir):
            for fname in files:
                if fname.endswith(".json"):
                    jpath = os.path.join(root, fname)
                    try:
                        with open(jpath, "r") as f:
                            meta = json.load(f)
                        stats["total_samples"] += 1
                        if meta.get("is_low_light", False) or meta.get("mode") == "night":
                            stats["night_samples"] += 1
                        else:
                            stats["day_samples"] += 1

                        if meta.get("label") == 1:
                            stats["open_eyes"] += 1
                        else:
                            stats["closed_eyes"] += 1

                        if meta.get("manual_tag", False):
                            stats["manual_feedback_samples"] += 1

                        cat = meta.get("category", "general")
                        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
                    except Exception:
                        continue

        return stats


class LiveAutoTrainer:
    """Incremental continuous trainer that trains model on live tester data."""

    def __init__(
        self,
        live_dir: str = "data/live_dataset",
        baseline_dir: str = "train_test/archive/Final/mrlEyes_2018_01",
        model_output_path: str = "train_test/models/low_light_eye_model.json"
    ):
        self.live_dir = live_dir
        self.baseline_dir = baseline_dir
        self.model_output_path = model_output_path
        self._is_training = False
        self._lock = threading.Lock()
        self.last_training_result: Dict[str, Any] = {}

    def is_training(self) -> bool:
        with self._lock:
            return self._is_training

    def train_async(
        self,
        epochs: int = 4,
        lr: float = 0.04,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> threading.Thread:
        """Start self-training in a background worker thread."""
        thread = threading.Thread(
            target=self._run_training_worker,
            args=(epochs, lr, on_complete),
            daemon=True
        )
        thread.start()
        return thread

    def _run_training_worker(
        self,
        epochs: int,
        lr: float,
        on_complete: Optional[Callable[[Dict[str, Any]], None]]
    ):
        with self._lock:
            self._is_training = True
        try:
            results = self.train_sync(epochs=epochs, lr=lr)
            self.last_training_result = results
            if on_complete:
                on_complete(results)
        finally:
            with self._lock:
                self._is_training = False

    def train_sync(self, epochs: int = 4, lr: float = 0.04) -> Dict[str, Any]:
        """Synchronously train the eye model with live collected samples."""
        t0 = time.time()
        live_samples = self._load_live_samples()
        
        # If live samples are limited, blend in baseline data
        baseline_samples = self._load_baseline_samples(max_samples=max(200, len(live_samples) * 2))
        combined_samples = live_samples + baseline_samples
        random.shuffle(combined_samples)

        if not combined_samples:
            return {
                "status": "skipped",
                "message": "No training samples found in live dataset or baseline.",
                "samples_trained": 0
            }

        # Initialize CNN model and load previous weights if present
        model = LowLightEyeCNN(input_shape=(64, 64, 1))
        if os.path.exists(self.model_output_path):
            try:
                model.load(self.model_output_path)
            except Exception as e:
                print(f"Could not load previous model weights: {e}")

        # Split into train and validation
        n_train = max(1, int(len(combined_samples) * 0.80))
        train_set = combined_samples[:n_train]
        val_set = combined_samples[n_train:]

        best_loss = float("inf")
        history = []

        batch_size = 16
        for epoch in range(1, epochs + 1):
            random.shuffle(train_set)
            total_loss = 0.0
            correct = 0
            total = 0

            for i in range(0, len(train_set), batch_size):
                batch = train_set[i:i + batch_size]
                batch_x, batch_y = [], []

                for sample in batch:
                    try:
                        img = cv2.imread(sample["path"], cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        if sample.get("is_low_light", False) and random.random() < 0.7:
                            img = apply_night_augmentation(img, simulate_extreme_dark=True)
                        
                        processed = preprocess_eye_sample(img, target_size=(64, 64), apply_clahe=True)
                        batch_x.append(processed)
                        batch_y.append(sample["label"])
                    except Exception:
                        continue

                if not batch_x:
                    continue

                X = np.array(batch_x, dtype=np.float32)
                Y = np.array(batch_y, dtype=np.float32)

                preds, d1, flat = model.forward_dense(X)
                eps = 1e-7
                loss = -np.mean(Y * np.log(preds + eps) + (1.0 - Y) * np.log(1.0 - preds + eps))

                # Backprop
                error = (preds - Y)[:, np.newaxis]
                grad_w_out = np.dot(d1.T, error) / float(len(Y))
                grad_b_out = np.mean(error)

                error_dense1 = np.dot(error, model.weights["w_out"].T) * (d1 > 0).astype(np.float32)
                grad_w_dense1 = np.dot(flat.T, error_dense1) / float(len(Y))
                grad_b_dense1 = np.mean(error_dense1, axis=0)

                model.weights["w_out"] -= lr * np.clip(grad_w_out, -0.5, 0.5)
                model.weights["b_out"] -= lr * float(np.clip(grad_b_out, -0.5, 0.5))
                model.weights["w_dense1"] -= (lr * 0.5) * np.clip(grad_w_dense1, -0.5, 0.5)
                model.weights["b_dense1"] -= (lr * 0.5) * np.clip(grad_b_dense1, -0.5, 0.5)

                total_loss += float(loss) * len(Y)
                preds_binary = (preds >= 0.50).astype(int)
                correct += int(np.sum(preds_binary == Y))
                total += len(Y)

            epoch_loss = total_loss / max(1, total)
            epoch_acc = correct / max(1, total)
            history.append({"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_acc})

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                model.save(self.model_output_path)

        elapsed = time.time() - t0
        res = {
            "status": "success",
            "samples_trained": len(combined_samples),
            "live_samples_count": len(live_samples),
            "baseline_samples_count": len(baseline_samples),
            "final_loss": history[-1]["loss"] if history else 0.0,
            "final_accuracy": history[-1]["accuracy"] if history else 0.0,
            "elapsed_seconds": round(elapsed, 2),
            "model_path": self.model_output_path
        }
        return res

    def _load_live_samples(self) -> List[Dict[str, Any]]:
        """Index all live collected eye image samples and metadata."""
        samples = []
        if not os.path.exists(self.live_dir):
            return samples

        for root, _, files in os.walk(self.live_dir):
            for fname in files:
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    img_path = os.path.join(root, fname)
                    json_path = os.path.splitext(img_path)[0] + ".json"
                    
                    label = 0 if "closed" in img_path.lower() or "lbl0" in fname else 1
                    is_low_light = "night" in img_path.lower()
                    
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r") as f:
                                meta = json.load(f)
                            label = meta.get("label", label)
                            is_low_light = meta.get("is_low_light", is_low_light)
                        except Exception:
                            pass
                            
                    samples.append({
                        "path": img_path,
                        "label": label,
                        "is_low_light": is_low_light,
                        "source": "live"
                    })
        return samples

    def _load_baseline_samples(self, max_samples: int = 300) -> List[Dict[str, Any]]:
        """Load a small subset from baseline MRL dataset for anchor stability."""
        samples = []
        if not os.path.exists(self.baseline_dir):
            return samples

        for root, _, files in os.walk(self.baseline_dir):
            for fname in files:
                if fname.endswith(".png"):
                    fpath = os.path.join(root, fname)
                    parts = os.path.splitext(fname)[0].split("_")
                    if len(parts) >= 8:
                        try:
                            samples.append({
                                "path": fpath,
                                "label": int(parts[4]),
                                "is_low_light": (int(parts[6]) == 0),
                                "source": "baseline"
                            })
                        except ValueError:
                            continue
                    if len(samples) >= max_samples:
                        break
            if len(samples) >= max_samples:
                break
        return samples
