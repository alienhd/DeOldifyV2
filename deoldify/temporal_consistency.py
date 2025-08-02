"""
Temporal Consistency Module for Video Colorization
Implements techniques to reduce flickering and improve frame-to-frame stability
Based on research from the provided PDF papers.
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import logging


class TemporalConsistencyProcessor:
    """Handles temporal consistency between video frames to reduce flickering."""
    
    def __init__(self, alpha: float = 0.7, flow_threshold: float = 0.5):
        """
        Initialize temporal consistency processor.
        
        Args:
            alpha: Blending factor for temporal consistency (0.0 = no consistency, 1.0 = maximum)
            flow_threshold: Threshold for optical flow magnitude to apply consistency
        """
        self.alpha = alpha
        self.flow_threshold = flow_threshold
        self.prev_frame = None
        self.prev_colored = None
        try:
            # Try newer OpenCV API first
            self.optical_flow = cv2.FarnebackOpticalFlow_create()
        except AttributeError:
            try:
                # Fallback to older DualTVL1 if available
                self.optical_flow = cv2.createOptFlow_DualTVL1()
            except AttributeError:
                # Final fallback - use basic Farneback method
                self.optical_flow = None
        
    def reset(self):
        """Reset the processor state for a new video sequence."""
        self.prev_frame = None
        self.prev_colored = None
        
    def process_frame(self, current_frame: Image.Image, current_colored: Image.Image) -> Image.Image:
        """
        Process a frame with temporal consistency.
        
        Args:
            current_frame: Original grayscale frame
            current_colored: Newly colorized frame
            
        Returns:
            Temporally consistent colorized frame
        """
        if self.prev_frame is None or self.prev_colored is None:
            # First frame - no consistency to apply
            self._update_previous(current_frame, current_colored)
            return current_colored
            
        try:
            # Apply temporal consistency
            consistent_frame = self._apply_temporal_consistency(
                current_frame, current_colored
            )
            self._update_previous(current_frame, consistent_frame)
            return consistent_frame
            
        except Exception as e:
            logging.warning(f"Temporal consistency failed: {e}, using original frame")
            self._update_previous(current_frame, current_colored)
            return current_colored
    
    def _apply_temporal_consistency(self, current_frame: Image.Image, current_colored: Image.Image) -> Image.Image:
        """Apply temporal consistency using optical flow and frame blending."""
        
        # Convert to numpy arrays
        curr_gray = np.array(current_frame.convert('L'))
        prev_gray = np.array(self.prev_frame.convert('L'))
        curr_colored_np = np.array(current_colored)
        prev_colored_np = np.array(self.prev_colored)
        
        # Calculate optical flow
        if self.optical_flow is not None:
            try:
                if hasattr(self.optical_flow, 'calc'):
                    flow = self.optical_flow.calc(prev_gray, curr_gray, None)
                else:
                    # Use basic Farneback optical flow as fallback
                    flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)[0]
                    if flow is None:
                        # If that fails too, use basic Farneback
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except Exception:
                # Fallback to basic Farneback method
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            # Use basic Farneback method
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Warp previous colored frame using optical flow
        h, w = curr_gray.shape
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.arange(w)
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]
        flow_map += flow
        
        warped_prev = cv2.remap(prev_colored_np, flow_map, None, cv2.INTER_LINEAR)
        
        # Calculate flow magnitude for adaptive blending
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # Create adaptive blending mask
        blend_mask = np.clip(flow_magnitude / self.flow_threshold, 0, 1)
        blend_mask = np.stack([blend_mask] * 3, axis=-1)
        
        # Adaptive blending based on flow magnitude
        alpha_adaptive = self.alpha * (1 - blend_mask)
        
        # Blend frames
        consistent_frame = (alpha_adaptive * warped_prev + 
                          (1 - alpha_adaptive) * curr_colored_np).astype(np.uint8)
        
        return Image.fromarray(consistent_frame)
    
    def _update_previous(self, frame: Image.Image, colored: Image.Image):
        """Update previous frame references."""
        self.prev_frame = frame.copy()
        self.prev_colored = colored.copy()


class EdgeEnhancementProcessor:
    """Handles edge enhancement to reduce fringing artifacts."""
    
    def __init__(self, edge_threshold: float = 0.1, enhancement_strength: float = 0.3):
        """
        Initialize edge enhancement processor.
        
        Args:
            edge_threshold: Threshold for edge detection
            enhancement_strength: Strength of edge enhancement
        """
        self.edge_threshold = edge_threshold
        self.enhancement_strength = enhancement_strength
    
    def enhance_edges(self, original: Image.Image, colored: Image.Image) -> Image.Image:
        """
        Enhance edges to reduce fringing artifacts.
        
        Args:
            original: Original grayscale image
            colored: Colorized image
            
        Returns:
            Edge-enhanced colorized image
        """
        try:
            # Convert to numpy arrays
            orig_np = np.array(original.convert('L'))
            colored_np = np.array(colored)
            
            # Detect edges in original image
            edges = cv2.Canny(orig_np, 50, 150)
            edges = edges.astype(np.float32) / 255.0
            
            # Create edge mask
            edge_mask = edges > self.edge_threshold
            
            # Convert original to RGB for blending
            orig_rgb = np.stack([orig_np] * 3, axis=-1)
            
            # Blend original edges with colored image
            enhanced = colored_np.copy().astype(np.float32)
            
            for c in range(3):
                enhanced[:, :, c] = np.where(
                    edge_mask,
                    (1 - self.enhancement_strength) * enhanced[:, :, c] + 
                    self.enhancement_strength * orig_rgb[:, :, c],
                    enhanced[:, :, c]
                )
            
            return Image.fromarray(enhanced.astype(np.uint8))
            
        except Exception as e:
            logging.warning(f"Edge enhancement failed: {e}, using original colored frame")
            return colored


class ColorStabilizer:
    """Stabilizes colors across frames using palette consistency."""
    
    def __init__(self, palette_size: int = 16, consistency_weight: float = 0.4):
        """
        Initialize color stabilizer.
        
        Args:
            palette_size: Number of dominant colors to track
            consistency_weight: Weight for palette consistency
        """
        self.palette_size = palette_size
        self.consistency_weight = consistency_weight
        self.reference_palette = None
        
    def reset(self):
        """Reset the stabilizer state."""
        self.reference_palette = None
    
    def stabilize_colors(self, colored_frame: Image.Image) -> Image.Image:
        """
        Stabilize colors using palette consistency.
        
        Args:
            colored_frame: Colorized frame to stabilize
            
        Returns:
            Color-stabilized frame
        """
        try:
            if self.reference_palette is None:
                # First frame - extract reference palette
                self.reference_palette = self._extract_palette(colored_frame)
                return colored_frame
            
            # Extract current palette
            current_palette = self._extract_palette(colored_frame)
            
            # Apply palette consistency
            stabilized = self._apply_palette_consistency(colored_frame, current_palette)
            
            # Update reference palette with exponential moving average
            self._update_reference_palette(current_palette)
            
            return stabilized
            
        except Exception as e:
            logging.warning(f"Color stabilization failed: {e}, using original frame")
            return colored_frame
    
    def _extract_palette(self, image: Image.Image) -> np.ndarray:
        """Extract dominant color palette from image."""
        # Convert to LAB color space for better color clustering
        img_np = np.array(image)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Reshape for clustering
        pixels = img_lab.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, self.palette_size, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers
    
    def _apply_palette_consistency(self, image: Image.Image, current_palette: np.ndarray) -> Image.Image:
        """Apply palette consistency to stabilize colors."""
        img_np = np.array(image)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Map current palette to reference palette
        palette_mapping = self._map_palettes(current_palette, self.reference_palette)
        
        # Apply color mapping
        h, w, _ = img_lab.shape
        img_flat = img_lab.reshape(-1, 3)
        
        # Find closest palette color for each pixel
        for i, current_color in enumerate(current_palette):
            target_color = palette_mapping[i]
            
            # Find pixels close to this palette color
            distances = np.linalg.norm(img_flat - current_color, axis=1)
            mask = distances < 50  # Threshold for color similarity
            
            # Blend towards target color
            if np.any(mask):
                blend_factor = self.consistency_weight
                img_flat[mask] = (1 - blend_factor) * img_flat[mask] + blend_factor * target_color
        
        # Convert back to RGB
        img_lab_final = img_flat.reshape(h, w, 3)
        img_rgb = cv2.cvtColor(img_lab_final.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_rgb)
    
    def _map_palettes(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Map source palette colors to closest target palette colors."""
        mapping = np.zeros_like(source)
        
        for i, src_color in enumerate(source):
            distances = np.linalg.norm(target - src_color, axis=1)
            closest_idx = np.argmin(distances)
            mapping[i] = target[closest_idx]
        
        return mapping
    
    def _update_reference_palette(self, current_palette: np.ndarray):
        """Update reference palette with exponential moving average."""
        alpha = 0.1  # Learning rate for palette update
        self.reference_palette = (1 - alpha) * self.reference_palette + alpha * current_palette