"""Image Preprocessing Module for Low-Light and Glare Robustness.

Analyzes frame illumination, applies adaptive CLAHE + Gamma enhancement
in low-light conditions, and mitigates specular glare reflections.
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np

def calculate_brightness(frame: np.ndarray) -> float:
    """Fast luminance calculation using downsampled proxy (< 0.05ms).
    
    Args:
        frame: OpenCV BGR image frame.
        
    Returns:
        float: Average luminance value in range [0.0, 255.0].
    """
    if frame is None or frame.size == 0:
        return 0.0
    small = cv2.resize(frame, (120, 90), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def apply_clahe_and_gamma(
    frame: np.ndarray,
    clip_limit: float = 2.5,
    grid_size: int = 8,
    gamma: float = 0.6,
    denoise: bool = False
) -> np.ndarray:
    """Enhance low-light image contrast via CLAHE on L-channel and Gamma adjustment.
    
    Args:
        frame: OpenCV BGR image frame.
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of neighborhood grid for histogram equalization.
        gamma: Gamma correction factor (< 1.0 brightens shadows).
        denoise: Whether to apply edge-preserving smoothing for low-light sensor noise.
        
    Returns:
        np.ndarray: Enhanced BGR image.
    """
    if frame is None or frame.size == 0:
        return frame
        
    work_frame = frame
    # Apply fast edge-preserving bilateral denoising only if necessary
    if denoise:
        work_frame = cv2.bilateralFilter(work_frame, d=3, sigmaColor=25, sigmaSpace=25)

    # 1. Convert to LAB color space
    lab = cv2.cvtColor(work_frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 2. Apply CLAHE to L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced_l = clahe.apply(l_channel)
    
    # 3. Apply Gamma correction to lift deep shadows if gamma != 1.0
    if gamma > 0 and gamma != 1.0:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_l = cv2.LUT(enhanced_l, table)
        
    # 4. Merge back and convert to BGR
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def reduce_specular_glare(frame: np.ndarray, threshold: int = 248) -> np.ndarray:
    """Zero-copy fast specular reflection dampening (sub-millisecond).
    
    Args:
        frame: OpenCV BGR image.
        threshold: Intensity threshold above which pixels are softened.
        
    Returns:
        np.ndarray: Glare-dampened image.
    """
    if frame is None or frame.size == 0:
        return frame
        
    # Check max brightness without full copy
    if int(frame.max()) < threshold:
        return frame
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if np.any(mask):
        out = frame.copy()
        out[mask] = np.clip(out[mask], 0, threshold)
        return out
    return frame

def enhance_eye_region(eye_img: np.ndarray, is_low_light: bool = True) -> np.ndarray:
    """Apply specialized localized enhancement to cropped eye regions in low/no light.
    
    Args:
        eye_img: BGR image crop of driver eye.
        is_low_light: Flag indicating low light condition.
        
    Returns:
        np.ndarray: Contrast-enhanced normalized eye image.
    """
    if eye_img is None or eye_img.size == 0:
        return eye_img
        
    if not is_low_light:
        return eye_img
        
    # Localized CLAHE for eye landmark / iris definition in the dark
    if len(eye_img.shape) == 3:
        gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_eye = eye_img
        
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_eye)
    if len(eye_img.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced

def preprocess_frame(
    frame: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, bool, float]:
    """Execute dynamic lighting analysis and adaptive enhancement.
    
    Args:
        frame: Raw input BGR frame from webcam.
        config: Loaded system configuration dictionary.
        
    Returns:
        Tuple: (processed_frame, is_low_light, luminance)
    """
    if frame is None or frame.size == 0:
        return frame, False, 0.0
        
    low_light_cfg = config.get("low_light", {})
    if not low_light_cfg.get("enabled", True):
        return frame, False, calculate_brightness(frame)
        
    threshold = low_light_cfg.get("brightness_threshold", 65.0)
    extreme_dark_threshold = low_light_cfg.get("extreme_dark_threshold", 30.0)
    clip_limit = low_light_cfg.get("clahe_clip_limit", 2.5)
    grid_size = low_light_cfg.get("clahe_grid_size", 8)
    base_gamma = low_light_cfg.get("gamma", 0.6)
    enable_denoise = low_light_cfg.get("denoise", True)
    
    luminance = calculate_brightness(frame)
    is_low_light = luminance < threshold
    
    output_frame = frame
    if is_low_light:
        # Scale gamma dynamically if in extreme darkness (luminance < 30)
        eff_gamma = base_gamma
        eff_clip = clip_limit
        if luminance < extreme_dark_threshold:
            # Extreme dark / no-light boost
            eff_gamma = max(0.35, base_gamma * (luminance / max(1.0, threshold)))
            eff_clip = min(4.0, clip_limit * 1.4)
            
        output_frame = apply_clahe_and_gamma(
            frame, 
            clip_limit=eff_clip, 
            grid_size=grid_size, 
            gamma=eff_gamma,
            denoise=enable_denoise and (luminance < extreme_dark_threshold)
        )
        
    # Global glare reduction is opt-in (default False) to avoid degrading normal frames;
    # targeted patch-level deglaring is applied to eye ROIs during detected glare runs.
    if low_light_cfg.get("global_glare_reduction", False):
        output_frame = reduce_specular_glare(output_frame)
    
    return output_frame, is_low_light, luminance
