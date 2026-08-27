"""Eyewear Detection and Optical Reflection Compensation Module.

Classifies driver ocular state into:
- NONE: Bare eyes without eyewear (standard direct eye tracking)
- REGULAR_GLASSES: Clear / prescription lenses with potential specular reflection/glare
- SUNGLASSES: Dark tinted / opaque lenses occluding direct eye visibility

Provides:
- Reflection-filtering & contrast restoration for clear glasses
- Ocular-to-skin luminance ratio and specular highlight analysis
- Temporal hysteresis voting to eliminate frame-to-frame classification jitter
"""

from enum import Enum
from collections import deque
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class EyewearType(Enum):
    """Enumeration of driver eyewear categories."""
    NONE = "NONE"
    REGULAR_GLASSES = "REGULAR_GLASSES"
    SUNGLASSES = "SUNGLASSES"


class EyewearDetector:
    """Detects eyewear type and provides optical through-reflection processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize eyewear detector with configuration parameters.
        
        Args:
            config: Loaded system configuration dictionary.
        """
        self.config = config or {}
        eyewear_cfg = self.config.get("eyewear", {})
        self.enabled = eyewear_cfg.get("enabled", True)
        self.sunglasses_lum_ratio_threshold = eyewear_cfg.get("sunglasses_lum_ratio_threshold", 0.52)
        self.glare_reflection_threshold = eyewear_cfg.get("glare_reflection_threshold", 218)
        self.glare_min_pixel_ratio = eyewear_cfg.get("glare_min_pixel_ratio", 0.025)
        self.glare_occlusion_ratio_threshold = eyewear_cfg.get("glare_occlusion_ratio_threshold", 0.15)
        self.glare_run_frames = eyewear_cfg.get("glare_run_frames", 4)
        self.glare_max_blind_frames = eyewear_cfg.get("glare_max_blind_frames", 60)
        self.hysteresis_frames = eyewear_cfg.get("hysteresis_frames", 15)
        
        # Glare run tracking state
        self.consecutive_glare_frames: int = 0
        self.is_eye_state_unknown: bool = False
        
        # 15-frame rolling vote queue to prevent jitter
        self.history = deque(maxlen=self.hysteresis_frames)
        self.current_eyewear_type = EyewearType.NONE
        self.current_confidence = 1.0

    def detect_eyewear(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        w: int,
        h: int
    ) -> Tuple[EyewearType, float, Dict[str, Any]]:
        """Classify driver eyewear state using facial landmarks and multi-feature analysis.
        
        Args:
            frame: Preprocessed BGR video frame.
            landmarks: 468 MediaPipe normalized landmark coordinates (shape (468, 2) or (468, 3)).
            w: Frame width in pixels.
            h: Frame height in pixels.
            
        Returns:
            Tuple of:
            - EyewearType (NONE, REGULAR_GLASSES, SUNGLASSES)
            - Classification confidence (0.0 to 1.0)
            - Telemetry dictionary with diagnostic metrics
        """
        if not self.enabled or frame is None or landmarks is None or len(landmarks) < 468:
            return self.current_eyewear_type, 1.0, {"reason": "landmarks_unavailable"}

        h_f, w_f = frame.shape[:2]

        try:
            # 1. Landmark index mappings
            # Left Eye: 362, 385, 387, 263, 373, 380; Eyebrow: 336, 296, 334, 293, 300
            # Right Eye: 33, 160, 158, 133, 153, 144; Eyebrow: 70, 63, 105, 66, 107
            # Cheek references: Left cheek 425, 280; Right cheek 205, 50
            # Nose bridge / Upper nose: 168, 6, 197, 195
            
            l_eye_pts = (landmarks[[362, 385, 387, 263, 373, 380], :2] * [w, h]).astype(int)
            r_eye_pts = (landmarks[[33, 160, 158, 133, 153, 144], :2] * [w, h]).astype(int)
            cheek_pts = (landmarks[[50, 205, 280, 425], :2] * [w, h]).astype(int)
            nose_bridge_pts = (landmarks[[168, 6, 197, 195], :2] * [w, h]).astype(int)

            # 2. Extract Eye Sockets ROI
            pad = 12
            lx1, ly1 = max(0, np.min(l_eye_pts[:, 0]) - pad), max(0, np.min(l_eye_pts[:, 1]) - pad)
            lx2, ly2 = min(w_f, np.max(l_eye_pts[:, 0]) + pad), min(h_f, np.max(l_eye_pts[:, 1]) + pad)

            rx1, ry1 = max(0, np.min(r_eye_pts[:, 0]) - pad), max(0, np.min(r_eye_pts[:, 1]) - pad)
            rx2, ry2 = min(w_f, np.max(r_eye_pts[:, 0]) + pad), min(h_f, np.max(r_eye_pts[:, 1]) + pad)

            if lx2 <= lx1 + 5 or ly2 <= ly1 + 5 or rx2 <= rx1 + 5 or ry2 <= ry1 + 5:
                return self.current_eyewear_type, self.current_confidence, {"reason": "roi_too_small"}

            l_crop = frame[ly1:ly2, lx1:lx2]
            r_crop = frame[ry1:ry2, rx1:rx2]

            # 3. Extract Cheek Skin Reference ROI (to compare bare skin brightness)
            cx1, cy1 = max(0, np.min(cheek_pts[:, 0]) - 8), max(0, np.min(cheek_pts[:, 1]) - 8)
            cx2, cy2 = min(w_f, np.max(cheek_pts[:, 0]) + 8), min(h_f, np.max(cheek_pts[:, 1]) + 8)
            
            if cx2 > cx1 + 5 and cy2 > cy1 + 5:
                cheek_crop = frame[cy1:cy2, cx1:cx2]
                cheek_lum = float(np.mean(cv2.cvtColor(cheek_crop, cv2.COLOR_BGR2GRAY)))
            else:
                cheek_lum = 120.0

            cheek_lum = max(15.0, cheek_lum)

            # Convert eye crops to grayscale
            l_gray = cv2.cvtColor(l_crop, cv2.COLOR_BGR2GRAY)
            r_gray = cv2.cvtColor(r_crop, cv2.COLOR_BGR2GRAY)

            l_lum = float(np.mean(l_gray))
            r_lum = float(np.mean(r_gray))
            eyes_lum = (l_lum + r_lum) / 2.0

            # Compute Ocular-to-Skin Luminance Ratio
            lum_ratio = eyes_lum / cheek_lum
            eye_contrast = (float(np.std(l_gray)) + float(np.std(r_gray))) / 2.0

            # 4. Specular Glare / Reflection Detection (characteristic of clear glass lenses)
            glare_mask_l = l_gray >= self.glare_reflection_threshold
            glare_mask_r = r_gray >= self.glare_reflection_threshold
            glare_ratio = float((np.sum(glare_mask_l) + np.sum(glare_mask_r)) / (l_gray.size + r_gray.size))

            # 5. Nose Bridge Frame Gradient Analysis
            bx1, by1 = max(0, np.min(nose_bridge_pts[:, 0]) - 6), max(0, np.min(nose_bridge_pts[:, 1]) - 6)
            bx2, by2 = min(w_f, np.max(nose_bridge_pts[:, 0]) + 6), min(h_f, np.max(nose_bridge_pts[:, 1]) + 6)
            frame_edge_detected = False
            if bx2 > bx1 + 4 and by2 > by1 + 4:
                bridge_gray = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
                bridge_grad = cv2.Sobel(bridge_gray, cv2.CV_32F, 1, 0, ksize=3)
                if float(np.max(np.abs(bridge_grad))) > 140.0:
                    frame_edge_detected = True

            # 6. Multi-Factor Classification Decision
            # Condition A: Dark Sunglasses (heavy tint, significantly darker than cheeks, low inner contrast)
            is_sunglasses = (lum_ratio < self.sunglasses_lum_ratio_threshold and eye_contrast < 22.0) or (lum_ratio < 0.40)
            
            # Condition B: Regular Clear Glasses (high specular glare spots, frame edge reflections, or moderate glare)
            is_regular_glasses = (glare_ratio >= self.glare_min_pixel_ratio) or (frame_edge_detected and glare_ratio > 0.008)

            if is_sunglasses:
                instant_type = EyewearType.SUNGLASSES
                conf = min(1.0, max(0.60, (self.sunglasses_lum_ratio_threshold - lum_ratio) * 3.0 + 0.60))
            elif is_regular_glasses:
                instant_type = EyewearType.REGULAR_GLASSES
                conf = min(1.0, max(0.65, glare_ratio * 15.0 + 0.65))
            else:
                instant_type = EyewearType.NONE
                conf = 0.85

            # 7. Temporal Smoothing via Rolling Voting Queue
            self.history.append(instant_type)
            # Find majority vote in window
            types_in_hist = list(self.history)
            voted_type = max(set(types_in_hist), key=types_in_hist.count)
            self.current_eyewear_type = voted_type
            self.current_confidence = conf

            telemetry = {
                "instant_type": instant_type.value,
                "voted_type": voted_type.value,
                "confidence": conf,
                "eyes_lum": eyes_lum,
                "cheek_lum": cheek_lum,
                "lum_ratio": lum_ratio,
                "eye_contrast": eye_contrast,
                "glare_ratio": glare_ratio,
                "frame_edge_detected": frame_edge_detected
            }
            return self.current_eyewear_type, self.current_confidence, telemetry

        except Exception as e:
            return self.current_eyewear_type, 0.50, {"error": str(e)}

    @staticmethod
    def filter_glass_reflection(eye_crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply targeted specular anti-glare filtering and multi-scale CLAHE to clear glasses crops.
        
        Enables the system to 'see through' lens reflections by dampening high-intensity glare
        blobs and amplifying pupil/iris edge gradients behind the glass.
        
        Args:
            eye_crop: BGR or Grayscale eye crop image.
            
        Returns:
            np.ndarray: Reflection-suppressed, contrast-restored eye image.
        """
        if eye_crop is None or eye_crop.size == 0:
            return eye_crop

        h, w = eye_crop.shape[:2]
        if h < 4 or w < 4:
            return eye_crop

        try:
            # 1. Convert to LAB for luminance channel isolation
            is_bgr = len(eye_crop.shape) == 3 and eye_crop.shape[2] == 3
            if is_bgr:
                lab = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2LAB)
                l_chan, a_chan, b_chan = cv2.split(lab)
            else:
                l_chan = eye_crop.copy()

            # 2. Identify specular reflection glare hotspots (L > 225)
            glare_mask = l_chan >= 225
            if np.any(glare_mask):
                # Dilate glare mask slightly to catch peripheral reflection halo
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dilated_mask = cv2.dilate(glare_mask.astype(np.uint8), kernel)
                
                # Inpaint or smooth glare spots using surrounding eye pixels
                l_chan = cv2.inpaint(l_chan, dilated_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

            # 3. Apply CLAHE on filtered luminance channel
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(4, 4))
            l_enhanced = clahe.apply(l_chan)

            if is_bgr:
                merged = cv2.merge((l_enhanced, a_chan, b_chan))
                return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return l_enhanced

        except Exception:
            return eye_crop

    def assess_eye_glare_occlusion(
        self,
        l_crop: Optional[np.ndarray],
        r_crop: Optional[np.ndarray]
    ) -> Tuple[bool, bool, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Assess localized eye patch quality and detect sustained glare saturation runs.
        
        When specular glare saturates the eye patch across consecutive frames, flags the frame
        as 'eye state unknown' to prevent corrupted EAR from polluting PERCLOS, and applies
        targeted anti-glare filtering specifically to the eye patches.
        
        Args:
            l_crop: Left eye BGR crop.
            r_crop: Right eye BGR crop.
            
        Returns:
            Tuple of:
            - is_glare_run (bool): True if sustained glare saturation run is active.
            - is_eye_state_unknown (bool): True if glare severely occludes eye recovery.
            - glare_ratio (float): Measured specular saturation ratio across eye crops.
            - filtered_l (Optional[np.ndarray]): Anti-glare filtered left eye crop.
            - filtered_r (Optional[np.ndarray]): Anti-glare filtered right eye crop.
        """
        if l_crop is None and r_crop is None:
            self.consecutive_glare_frames = 0
            self.is_eye_state_unknown = False
            return False, False, 0.0, l_crop, r_crop

        try:
            total_pixels = 0
            glare_pixels = 0

            for crop in (l_crop, r_crop):
                if crop is not None and crop.size > 0:
                    if len(crop.shape) == 3:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = crop
                    total_pixels += gray.size
                    glare_pixels += int(np.sum(gray >= self.glare_reflection_threshold))

            if total_pixels == 0:
                self.consecutive_glare_frames = 0
                self.is_eye_state_unknown = False
                return False, False, 0.0, l_crop, r_crop

            glare_ratio = float(glare_pixels / total_pixels)

            # Check if current frame has high localized glare saturation
            is_patch_saturated = glare_ratio >= self.glare_occlusion_ratio_threshold

            if is_patch_saturated:
                self.consecutive_glare_frames += 1
            else:
                self.consecutive_glare_frames = max(0, self.consecutive_glare_frames - 1)

            is_glare_run = self.consecutive_glare_frames >= self.glare_run_frames
            out_l = l_crop
            out_r = r_crop

            # Only apply patch-specific deglaring when a genuine glare condition/run is detected
            if is_glare_run or is_patch_saturated:
                if l_crop is not None:
                    out_l = self.filter_glass_reflection(l_crop)
                if r_crop is not None:
                    out_r = self.filter_glass_reflection(r_crop)

                # Re-check post-filtering residual glare
                res_glare = 0
                res_total = 0
                for post_crop in (out_l, out_r):
                    if post_crop is not None and post_crop.size > 0:
                        g = cv2.cvtColor(post_crop, cv2.COLOR_BGR2GRAY) if len(post_crop.shape) == 3 else post_crop
                        res_total += g.size
                        res_glare += int(np.sum(g >= self.glare_reflection_threshold))
                
                post_glare_ratio = float(res_glare / max(1, res_total))
                # If post-filtering glare remains excessively high, eye landmarks cannot be trusted
                if post_glare_ratio >= (self.glare_occlusion_ratio_threshold * 0.8) and is_glare_run:
                    self.is_eye_state_unknown = True
                else:
                    self.is_eye_state_unknown = False
            else:
                self.is_eye_state_unknown = False

            return is_glare_run, self.is_eye_state_unknown, glare_ratio, out_l, out_r

        except Exception:
            return False, False, 0.0, l_crop, r_crop

