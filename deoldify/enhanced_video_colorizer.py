"""
Enhanced Video Colorization Module
Integrates temporal consistency, advanced post-processing, and multiple colorization methods.
"""
import os
import shutil
import logging
from pathlib import Path
from PIL import Image
import ffmpeg
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .temporal_consistency import TemporalConsistencyProcessor, EdgeEnhancementProcessor, ColorStabilizer
from .enhanced_post_processing import PostProcessingPipeline
from .visualize import VideoColorizer, ModelImageVisualizer
from .filters import IFilter, MasterFilter, ColorizerFilter


class EnhancedVideoColorizer(VideoColorizer):
    """Enhanced video colorizer with temporal consistency and advanced post-processing."""
    
    def __init__(self, vis: ModelImageVisualizer, enable_temporal_consistency: bool = True,
                 enable_edge_enhancement: bool = True, enable_color_stabilization: bool = True):
        """
        Initialize enhanced video colorizer.
        
        Args:
            vis: Model image visualizer
            enable_temporal_consistency: Enable temporal consistency processing
            enable_edge_enhancement: Enable edge enhancement
            enable_color_stabilization: Enable color stabilization
        """
        super().__init__(vis)
        
        # Enhanced processing components
        self.temporal_processor = TemporalConsistencyProcessor() if enable_temporal_consistency else None
        self.edge_processor = EdgeEnhancementProcessor() if enable_edge_enhancement else None
        self.color_stabilizer = ColorStabilizer() if enable_color_stabilization else None
        self.post_processor = PostProcessingPipeline()
        
        # Processing parameters
        self.post_processing_preset = "balanced"
        self.custom_post_processing = {}
        
    def set_post_processing_preset(self, preset: str):
        """Set post-processing preset."""
        self.post_processing_preset = preset
        
    def set_custom_post_processing(self, params: Dict[str, Any]):
        """Set custom post-processing parameters."""
        self.custom_post_processing = params
    
    def colorize_from_file_name_enhanced(
        self, 
        file_name: str, 
        render_factor: int = None,
        watermarked: bool = True, 
        post_process: bool = True,
        temporal_consistency: bool = True,
        edge_enhancement: bool = True,
        color_stabilization: bool = True,
        post_processing_preset: str = "balanced",
        frame_skip: int = 1
    ) -> Path:
        """
        Enhanced video colorization with all improvements.
        
        Args:
            file_name: Source video file name
            render_factor: Render quality factor
            watermarked: Add watermark
            post_process: Apply post-processing
            temporal_consistency: Apply temporal consistency
            edge_enhancement: Apply edge enhancement
            color_stabilization: Apply color stabilization
            post_processing_preset: Post-processing preset
            frame_skip: Process every Nth frame (1 = all frames)
            
        Returns:
            Path to output video
        """
        source_path = self.source_folder / file_name
        
        # Set processing options
        self.post_processing_preset = post_processing_preset
        
        return self._colorize_from_path_enhanced(
            source_path, 
            render_factor=render_factor,
            watermarked=watermarked,
            post_process=post_process,
            temporal_consistency=temporal_consistency,
            edge_enhancement=edge_enhancement,
            color_stabilization=color_stabilization,
            frame_skip=frame_skip
        )
    
    def _colorize_from_path_enhanced(
        self, 
        source_path: Path, 
        render_factor: int = None,
        watermarked: bool = True, 
        post_process: bool = True,
        temporal_consistency: bool = True,
        edge_enhancement: bool = True,
        color_stabilization: bool = True,
        frame_skip: int = 1
    ) -> Path:
        """Enhanced colorization from path with all processing options."""
        
        if not source_path.exists():
            raise Exception(f'Video at path {source_path} could not be found.')
        
        # Reset processors for new video
        if self.temporal_processor and temporal_consistency:
            self.temporal_processor.reset()
        if self.color_stabilizer and color_stabilization:
            self.color_stabilizer.reset()
        
        # Extract frames
        self._extract_raw_frames(source_path)
        
        # Enhanced frame colorization
        self._colorize_raw_frames_enhanced(
            source_path, 
            render_factor=render_factor,
            post_process=post_process,
            watermarked=watermarked,
            temporal_consistency=temporal_consistency,
            edge_enhancement=edge_enhancement,
            color_stabilization=color_stabilization,
            frame_skip=frame_skip
        )
        
        # Build output video
        return self._build_video(source_path)
    
    def _colorize_raw_frames_enhanced(
        self,
        source_path: Path,
        render_factor: int = None,
        post_process: bool = True,
        watermarked: bool = True,
        temporal_consistency: bool = True,
        edge_enhancement: bool = True,
        color_stabilization: bool = True,
        frame_skip: int = 1
    ):
        """Enhanced frame colorization with temporal consistency and post-processing."""
        
        colorframes_folder = self.colorframes_root / source_path.stem
        colorframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(colorframes_folder)
        
        bwframes_folder = self.bwframes_root / source_path.stem
        frame_files = sorted([f for f in os.listdir(str(bwframes_folder)) if f.endswith('.jpg')])
        
        logging.info(f"Processing {len(frame_files)} frames with enhanced colorization")
        
        for i, img_name in enumerate(frame_files):
            # Skip frames if frame_skip > 1
            if i % frame_skip != 0:
                # Copy previous frame for skipped frames
                if i > 0:
                    prev_frame_name = frame_files[i-1] if (i-1) % frame_skip == 0 else frame_files[i - (i % frame_skip)]
                    prev_colored_path = colorframes_folder / prev_frame_name
                    if prev_colored_path.exists():
                        shutil.copy(str(prev_colored_path), str(colorframes_folder / img_name))
                continue
            
            img_path = bwframes_folder / img_name
            
            if not os.path.isfile(str(img_path)):
                continue
            
            try:
                # Load original frame
                original_frame = Image.open(str(img_path)).convert('RGB')
                
                # Basic colorization
                colored_image = self.vis.get_transformed_image(
                    str(img_path), 
                    render_factor=render_factor, 
                    post_process=False,  # We'll handle post-processing separately
                    watermarked=False    # We'll handle watermarking later
                )
                
                # Apply enhanced processing pipeline
                if edge_enhancement and self.edge_processor:
                    colored_image = self.edge_processor.enhance_edges(original_frame, colored_image)
                
                if temporal_consistency and self.temporal_processor:
                    colored_image = self.temporal_processor.process_frame(original_frame, colored_image)
                
                if color_stabilization and self.color_stabilizer:
                    colored_image = self.color_stabilizer.stabilize_colors(colored_image)
                
                # Apply post-processing
                if post_process:
                    if self.custom_post_processing:
                        colored_image = self._apply_custom_post_processing(colored_image)
                    else:
                        colored_image = self.post_processor.apply_enhancement_preset(
                            colored_image, self.post_processing_preset
                        )
                
                # Apply watermark if requested
                if watermarked:
                    from .visualize import get_watermarked
                    colored_image = get_watermarked(colored_image)
                
                # Save processed frame
                colored_image.save(str(colorframes_folder / img_name))
                
                if i % 10 == 0:
                    logging.info(f"Processed frame {i+1}/{len(frame_files)}")
                    
            except Exception as e:
                logging.error(f"Error processing frame {img_name}: {e}")
                # Save original frame as fallback
                original_frame = Image.open(str(img_path)).convert('RGB')
                original_frame.save(str(colorframes_folder / img_name))
    
    def _apply_custom_post_processing(self, image: Image.Image) -> Image.Image:
        """Apply custom post-processing parameters."""
        processed = image
        
        try:
            # Apply custom parameters
            if 'bilateral_filter' in self.custom_post_processing:
                params = self.custom_post_processing['bilateral_filter']
                processed = self.post_processor.processor.apply_bilateral_filter(
                    processed, **params
                )
            
            if 'saturation' in self.custom_post_processing:
                factor = self.custom_post_processing['saturation']
                processed = self.post_processor.processor.enhance_saturation(processed, factor)
            
            if 'contrast' in self.custom_post_processing:
                factor = self.custom_post_processing['contrast']
                processed = self.post_processor.processor.enhance_contrast(processed, factor)
            
            if 'unsharp_mask' in self.custom_post_processing:
                params = self.custom_post_processing['unsharp_mask']
                processed = self.post_processor.processor.apply_unsharp_mask(
                    processed, **params
                )
            
            if 'color_balance' in self.custom_post_processing:
                params = self.custom_post_processing['color_balance']
                processed = self.post_processor.processor.apply_color_balance(
                    processed, **params
                )
            
            if 'clahe' in self.custom_post_processing:
                params = self.custom_post_processing['clahe']
                processed = self.post_processor.processor.apply_clahe(processed, **params)
            
        except Exception as e:
            logging.warning(f"Custom post-processing failed: {e}")
            
        return processed


class MultiModelVideoColorizer:
    """Video colorizer supporting multiple models and methods."""
    
    def __init__(self):
        """Initialize multi-model video colorizer."""
        self.colorizers = {}
        self.current_method = "stable"
        
    def add_colorizer(self, name: str, colorizer: EnhancedVideoColorizer):
        """Add a colorizer method."""
        self.colorizers[name] = colorizer
        
    def set_method(self, method: str):
        """Set the current colorization method."""
        if method in self.colorizers:
            self.current_method = method
        else:
            logging.warning(f"Method {method} not available, using {self.current_method}")
    
    def get_available_methods(self) -> list:
        """Get list of available colorization methods."""
        return list(self.colorizers.keys())
    
    def colorize_video(self, file_name: str, method: str = None, **kwargs) -> Path:
        """
        Colorize video using specified method.
        
        Args:
            file_name: Source video file name
            method: Colorization method to use
            **kwargs: Additional parameters for colorization
            
        Returns:
            Path to output video
        """
        if method:
            self.set_method(method)
        
        if self.current_method not in self.colorizers:
            raise ValueError(f"No colorizer available for method: {self.current_method}")
        
        return self.colorizers[self.current_method].colorize_from_file_name_enhanced(
            file_name, **kwargs
        )