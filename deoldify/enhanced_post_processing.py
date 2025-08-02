"""
Enhanced Post-Processing Module for DeOldify
Provides advanced post-processing techniques for better colorization results.
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Optional
import logging


class EnhancedPostProcessor:
    """Enhanced post-processing with multiple filtering and enhancement options."""
    
    def __init__(self):
        """Initialize enhanced post-processor."""
        pass
    
    def apply_bilateral_filter(self, image: Image.Image, d: int = 9, 
                              sigma_color: float = 75, sigma_space: float = 75) -> Image.Image:
        """
        Apply bilateral filter for noise reduction while preserving edges.
        
        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image
        """
        try:
            img_array = np.array(image)
            filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
            return Image.fromarray(filtered)
        except Exception as e:
            logging.warning(f"Bilateral filter failed: {e}")
            return image
    
    def apply_guided_filter(self, image: Image.Image, guide: Optional[Image.Image] = None, 
                           radius: int = 8, eps: float = 0.01) -> Image.Image:
        """
        Apply guided filter for edge-preserving smoothing.
        
        Args:
            image: Input image to filter
            guide: Guide image (uses input if None)
            radius: Radius of the filter
            eps: Regularization parameter
            
        Returns:
            Filtered image
        """
        try:
            img_array = np.array(image).astype(np.float32) / 255.0
            guide_array = np.array(guide if guide else image).astype(np.float32) / 255.0
            
            if len(guide_array.shape) == 3:
                guide_array = cv2.cvtColor(guide_array, cv2.COLOR_RGB2GRAY)
            
            filtered_channels = []
            for c in range(img_array.shape[2]):
                filtered_c = self._guided_filter_single(guide_array, img_array[:, :, c], radius, eps)
                filtered_channels.append(filtered_c)
            
            filtered = np.stack(filtered_channels, axis=2)
            filtered = np.clip(filtered * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(filtered)
        except Exception as e:
            logging.warning(f"Guided filter failed: {e}")
            return image
    
    def _guided_filter_single(self, guide: np.ndarray, src: np.ndarray, 
                             radius: int, eps: float) -> np.ndarray:
        """Apply guided filter to single channel."""
        mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        return mean_a * guide + mean_b
    
    def enhance_saturation(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Enhance color saturation.
        
        Args:
            image: Input image
            factor: Saturation enhancement factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        try:
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logging.warning(f"Saturation enhancement failed: {e}")
            return image
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """
        Enhance contrast.
        
        Args:
            image: Input image
            factor: Contrast enhancement factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logging.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def apply_unsharp_mask(self, image: Image.Image, radius: float = 2.0, 
                          amount: float = 1.5, threshold: int = 0) -> Image.Image:
        """
        Apply unsharp mask for edge enhancement.
        
        Args:
            image: Input image
            radius: Blur radius
            amount: Enhancement amount
            threshold: Threshold for enhancement
            
        Returns:
            Enhanced image
        """
        try:
            img_array = np.array(image).astype(np.float32)
            
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
            
            # Create unsharp mask
            unsharp = img_array + amount * (img_array - blurred)
            
            # Apply threshold
            if threshold > 0:
                diff = np.abs(img_array - blurred)
                mask = diff >= threshold
                unsharp = np.where(mask, unsharp, img_array)
            
            unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
            return Image.fromarray(unsharp)
        except Exception as e:
            logging.warning(f"Unsharp mask failed: {e}")
            return image
    
    def apply_clahe(self, image: Image.Image, clip_limit: float = 2.0, 
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of the neighborhood for equalization
            
        Returns:
            Enhanced image
        """
        try:
            img_array = np.array(image)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(enhanced)
        except Exception as e:
            logging.warning(f"CLAHE failed: {e}")
            return image
    
    def apply_color_balance(self, image: Image.Image, temperature: float = 0.0, 
                           tint: float = 0.0) -> Image.Image:
        """
        Apply color temperature and tint adjustments.
        
        Args:
            image: Input image
            temperature: Color temperature adjustment (-1.0 to 1.0)
            tint: Tint adjustment (-1.0 to 1.0)
            
        Returns:
            Color-balanced image
        """
        try:
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Temperature adjustment (blue-yellow)
            if temperature != 0:
                if temperature > 0:  # Warmer
                    img_array[:, :, 0] *= (1 + temperature * 0.3)  # More red
                    img_array[:, :, 1] *= (1 + temperature * 0.1)  # Slightly more green
                    img_array[:, :, 2] *= (1 - temperature * 0.2)  # Less blue
                else:  # Cooler
                    img_array[:, :, 0] *= (1 + temperature * 0.2)  # Less red
                    img_array[:, :, 1] *= (1 + temperature * 0.1)  # Slightly less green
                    img_array[:, :, 2] *= (1 - temperature * 0.3)  # More blue
            
            # Tint adjustment (green-magenta)
            if tint != 0:
                if tint > 0:  # More green
                    img_array[:, :, 1] *= (1 + tint * 0.2)
                    img_array[:, :, 0] *= (1 - tint * 0.1)
                    img_array[:, :, 2] *= (1 - tint * 0.1)
                else:  # More magenta
                    img_array[:, :, 1] *= (1 + tint * 0.2)
                    img_array[:, :, 0] *= (1 - tint * 0.1)
                    img_array[:, :, 2] *= (1 - tint * 0.1)
            
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        except Exception as e:
            logging.warning(f"Color balance failed: {e}")
            return image
    
    def apply_selective_color_adjustment(self, image: Image.Image, 
                                       color_adjustments: dict) -> Image.Image:
        """
        Apply selective color adjustments to specific color ranges.
        
        Args:
            image: Input image
            color_adjustments: Dict with color ranges and adjustment factors
                              Format: {'reds': (hue_shift, sat_mult, val_mult), ...}
            
        Returns:
            Adjusted image
        """
        try:
            img_array = np.array(image)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            color_ranges = {
                'reds': [(0, 10), (170, 180)],
                'oranges': [(10, 25)],
                'yellows': [(25, 35)],
                'greens': [(35, 85)],
                'cyans': [(85, 95)],
                'blues': [(95, 125)],
                'magentas': [(125, 170)]
            }
            
            for color_name, adjustments in color_adjustments.items():
                if color_name not in color_ranges:
                    continue
                
                hue_shift, sat_mult, val_mult = adjustments
                ranges = color_ranges[color_name]
                
                for hue_range in ranges:
                    mask = np.zeros(hsv.shape[:2], dtype=bool)
                    if len(hue_range) == 2:
                        mask |= (hsv[:, :, 0] >= hue_range[0]) & (hsv[:, :, 0] <= hue_range[1])
                    
                    if np.any(mask):
                        hsv[mask, 0] = (hsv[mask, 0] + hue_shift) % 180
                        hsv[mask, 1] = np.clip(hsv[mask, 1] * sat_mult, 0, 255)
                        hsv[mask, 2] = np.clip(hsv[mask, 2] * val_mult, 0, 255)
            
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return Image.fromarray(rgb)
        except Exception as e:
            logging.warning(f"Selective color adjustment failed: {e}")
            return image


class PostProcessingPipeline:
    """Pipeline for applying multiple post-processing effects."""
    
    def __init__(self):
        """Initialize post-processing pipeline."""
        self.processor = EnhancedPostProcessor()
    
    def apply_enhancement_preset(self, image: Image.Image, preset: str = "balanced") -> Image.Image:
        """
        Apply a preset combination of enhancements.
        
        Args:
            image: Input image
            preset: Enhancement preset name
            
        Returns:
            Enhanced image
        """
        presets = {
            "balanced": self._balanced_preset,
            "vivid": self._vivid_preset,
            "soft": self._soft_preset,
            "sharp": self._sharp_preset,
            "cinematic": self._cinematic_preset
        }
        
        if preset not in presets:
            preset = "balanced"
        
        return presets[preset](image)
    
    def _balanced_preset(self, image: Image.Image) -> Image.Image:
        """Apply balanced enhancement preset."""
        # Bilateral filter for noise reduction
        enhanced = self.processor.apply_bilateral_filter(image, d=5, sigma_color=50, sigma_space=50)
        
        # Slight contrast enhancement
        enhanced = self.processor.enhance_contrast(enhanced, 1.05)
        
        # Slight saturation boost
        enhanced = self.processor.enhance_saturation(enhanced, 1.1)
        
        # Gentle unsharp mask
        enhanced = self.processor.apply_unsharp_mask(enhanced, radius=1.0, amount=0.5)
        
        return enhanced
    
    def _vivid_preset(self, image: Image.Image) -> Image.Image:
        """Apply vivid enhancement preset."""
        # Strong saturation enhancement
        enhanced = self.processor.enhance_saturation(image, 1.3)
        
        # Contrast enhancement
        enhanced = self.processor.enhance_contrast(enhanced, 1.15)
        
        # CLAHE for local contrast
        enhanced = self.processor.apply_clahe(enhanced, clip_limit=3.0, tile_grid_size=(8, 8))
        
        # Unsharp mask for sharpness
        enhanced = self.processor.apply_unsharp_mask(enhanced, radius=1.5, amount=1.0)
        
        return enhanced
    
    def _soft_preset(self, image: Image.Image) -> Image.Image:
        """Apply soft enhancement preset."""
        # Strong bilateral filter for smoothing
        enhanced = self.processor.apply_bilateral_filter(image, d=15, sigma_color=80, sigma_space=80)
        
        # Guided filter for additional smoothing
        enhanced = self.processor.apply_guided_filter(enhanced, radius=4, eps=0.02)
        
        # Slight saturation reduction for softer look
        enhanced = self.processor.enhance_saturation(enhanced, 0.95)
        
        return enhanced
    
    def _sharp_preset(self, image: Image.Image) -> Image.Image:
        """Apply sharp enhancement preset."""
        # Strong unsharp mask
        enhanced = self.processor.apply_unsharp_mask(image, radius=2.0, amount=2.0)
        
        # Contrast enhancement
        enhanced = self.processor.enhance_contrast(enhanced, 1.2)
        
        # Bilateral filter to reduce noise from sharpening
        enhanced = self.processor.apply_bilateral_filter(enhanced, d=3, sigma_color=30, sigma_space=30)
        
        return enhanced
    
    def _cinematic_preset(self, image: Image.Image) -> Image.Image:
        """Apply cinematic enhancement preset."""
        # Color temperature adjustment (slightly warm)
        enhanced = self.processor.apply_color_balance(image, temperature=0.1, tint=-0.05)
        
        # Selective color adjustments for cinematic look
        color_adjustments = {
            'reds': (0, 1.1, 0.95),      # Slightly enhance reds, darken
            'blues': (-2, 1.2, 0.9),     # Cool blues, enhance saturation
            'yellows': (2, 0.9, 1.05)    # Warm yellows, brighten
        }
        enhanced = self.processor.apply_selective_color_adjustment(enhanced, color_adjustments)
        
        # Contrast enhancement
        enhanced = self.processor.enhance_contrast(enhanced, 1.1)
        
        # Gentle unsharp mask
        enhanced = self.processor.apply_unsharp_mask(enhanced, radius=1.2, amount=0.8)
        
        return enhanced