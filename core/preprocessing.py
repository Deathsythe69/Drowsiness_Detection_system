"""Image Preprocessing Module for Low-Light and Glare Robustness.

Analyzes frame illumination, applies adaptive CLAHE + Gamma enhancement
in low-light conditions, and mitigates specular glare reflections.
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np

def calculate_brightness(frame: np.ndarray) -> float:
    """Calculate the average perceived brightness (luminance) of a frame.
    
    Args:
        frame: OpenCV BGR image frame.
        
    Returns:
        float: Average luminance value in range [0.0, 255.0].
    """
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def apply_clahe_and_gamma(
    frame: np.ndarray,
    clip_limit: float = 2.5,
    grid_size: int = 8,
    gamma: float = 0.6
) -> np.ndarray:
    """Enhance low-light image contrast via CLAHE on L-channel and Gamma adjustment.
    
    Args:
        frame: OpenCV BGR image frame.
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of neighborhood grid for histogram equalization.
        gamma: Gamma correction factor (< 1.0 brightens shadows).
        
    Returns:
        np.ndarray: Enhanced BGR image.
    """
    if frame is None or frame.size == 0:
        return frame
        
    # 1. Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
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

def reduce_specular_glare(frame: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Fast, lightweight specular reflection dampening (low CPU / low latency).
    
    Args:
        frame: OpenCV BGR image.
        threshold: Intensity threshold above which pixels are softened.
        
    Returns:
        np.ndarray: Glare-dampened image.
    """
    if frame is None or frame.size == 0:
        return frame
        
    # Fast threshold clipping without heavy inpainting CPU overhead
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if np.any(mask):
        frame = frame.copy()
        # Soften extreme saturated highlights without blurring the whole face
        frame[mask] = np.clip(frame[mask], 0, threshold)
    return frame

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
    clip_limit = low_light_cfg.get("clahe_clip_limit", 2.5)
    grid_size = low_light_cfg.get("clahe_grid_size", 8)
    gamma = low_light_cfg.get("gamma", 0.6)
    
    luminance = calculate_brightness(frame)
    is_low_light = luminance < threshold
    
    output_frame = frame
    if is_low_light:
        output_frame = apply_clahe_and_gamma(frame, clip_limit=clip_limit, grid_size=grid_size, gamma=gamma)
        
    # Also reduce harsh reflection hotspots on glasses
    output_frame = reduce_specular_glare(output_frame)
    
    return output_frame, is_low_light, luminance
