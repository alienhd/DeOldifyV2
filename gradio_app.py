"""
Comprehensive Gradio UI for Enhanced DeOldify Video Colorization
Provides an intuitive interface for video colorization with advanced options.
"""
import gradio as gr
import os
import tempfile
import shutil
from pathlib import Path
import logging
import traceback
from typing import Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model downloader
from model_downloader import ModelDownloader, MODEL_CONFIGS

# Import DeOldify components
try:
    from deoldify.enhanced_video_colorizer import EnhancedVideoColorizer, MultiModelVideoColorizer
    from deoldify.visualize import get_stable_video_colorizer, get_artistic_video_colorizer, get_video_colorizer
    from deoldify.enhanced_post_processing import PostProcessingPipeline
    from deoldify import device
    from deoldify.device_id import DeviceId
    
    # Set device (prefer GPU, fallback to CPU with proper logging)
    try:
        # Try GPU first
        device.set(device=DeviceId.GPU0)
        device_info = device.get_device_info()
        logger.info(f"Device setup complete: {device_info}")
        if device_info['is_gpu']:
            logger.info(f"Using GPU: {device_info.get('cuda_device_name', 'Unknown GPU')}")
        else:
            logger.info("Using CPU (GPU not available)")
    except Exception as e:
        logger.error(f"Error setting up device: {e}")
        device.set(device=DeviceId.CPU)
        device_info = device.get_device_info()
        logger.info(f"Fallback to CPU: {device_info}")
    
    DEOLDIFY_AVAILABLE = True
except ImportError as e:
    logger.error(f"DeOldify components not available: {e}")
    DEOLDIFY_AVAILABLE = False


class GradioVideoColorizer:
    """Gradio interface wrapper for enhanced video colorization."""
    
    def __init__(self):
        """Initialize Gradio video colorizer."""
        self.output_dir = Path("./gradio_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize model downloader
        self.model_downloader = ModelDownloader()
        
        # Check model availability and setup
        self.models_available = False
        self.models_status = {}
        self.multi_colorizer = None
        
        if DEOLDIFY_AVAILABLE:
            self.check_and_setup_models()
        
        self.post_processor = PostProcessingPipeline() if DEOLDIFY_AVAILABLE else None
        
    def check_and_setup_models(self):
        """Check model availability and download if needed."""
        try:
            # Check which models are missing
            missing_models = self.model_downloader.get_missing_models()
            self.models_status = self.model_downloader.verify_models()
            
            if missing_models:
                logger.warning(f"Missing models: {missing_models}")
                self.models_available = False
            else:
                logger.info("All required models are available")
                self.models_available = True
                self.setup_colorizers()
                
        except Exception as e:
            logger.error(f"Error checking models: {e}")
            self.models_available = False
    
    def download_missing_models(self, progress=None):
        """Download missing models with progress feedback."""
        try:
            if progress:
                progress(0.1, desc="Checking missing models...")
            
            missing_models = self.model_downloader.get_missing_models()
            
            if not missing_models:
                if progress:
                    progress(1.0, desc="All models already available!")
                return "All models are already available", self.get_model_status_info()
            
            def progress_callback(pct, desc):
                if progress:
                    progress(0.1 + 0.8 * pct, desc=desc)
            
            if progress:
                progress(0.2, desc=f"Downloading {len(missing_models)} models...")
            
            results = self.model_downloader.download_all_models(progress_callback)
            
            success_count = sum(results.values())
            total_count = len(results)
            
            if success_count == total_count:
                if progress:
                    progress(0.95, desc="Setting up colorizers...")
                self.models_available = True
                self.setup_colorizers()
                if progress:
                    progress(1.0, desc="Models downloaded and ready!")
                return f"Successfully downloaded all {total_count} models", self.get_model_status_info()
            else:
                failed_models = [name for name, success in results.items() if not success]
                error_msg = f"Failed to download {len(failed_models)} models: {failed_models}"
                return error_msg, self.get_model_status_info()
                
        except Exception as e:
            error_msg = f"Error downloading models: {str(e)}"
            logger.error(error_msg)
            return error_msg, self.get_model_status_info()
    def setup_colorizers(self):
        """Setup different colorization models."""
        if not self.models_available:
            logger.warning("Cannot setup colorizers - models not available")
            return False
            
        try:
            # Initialize colorizers
            video_vis = get_video_colorizer(render_factor=21)
            stable_vis = get_stable_video_colorizer(render_factor=21)
            artistic_vis = get_artistic_video_colorizer(render_factor=35)
            
            # Create enhanced colorizers
            self.multi_colorizer = MultiModelVideoColorizer()
            
            video_enhanced = EnhancedVideoColorizer(
                video_vis.vis,
                enable_temporal_consistency=True,
                enable_edge_enhancement=True,
                enable_color_stabilization=True
            )
            
            stable_enhanced = EnhancedVideoColorizer(
                stable_vis.vis,
                enable_temporal_consistency=True,
                enable_edge_enhancement=True,
                enable_color_stabilization=True
            )
            
            artistic_enhanced = EnhancedVideoColorizer(
                artistic_vis.vis,
                enable_temporal_consistency=True,
                enable_edge_enhancement=True,
                enable_color_stabilization=True
            )
            
            self.multi_colorizer.add_colorizer("video", video_enhanced)
            self.multi_colorizer.add_colorizer("stable", stable_enhanced)
            self.multi_colorizer.add_colorizer("artistic", artistic_enhanced)
            
            logger.info("Colorizers initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup colorizers: {e}")
            self.multi_colorizer = None
            self.models_available = False
            return False
    
    def validate_video_file(self, video_path: str):
        """
        Validate video file for compatibility.
        
        Returns:
            tuple: (is_valid, message)
        """
        try:
            video_path = Path(video_path)
            
            # Check if file exists
            if not video_path.exists():
                return False, "Video file not found"
            
            # Check file extension
            allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            if video_path.suffix.lower() not in allowed_extensions:
                return False, f"Unsupported format. Use: {', '.join(allowed_extensions)}"
            
            # Check file size - Allow larger files for HD processing
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 2000:  # 2GB limit instead of 500MB
                return False, f"File too large ({file_size_mb:.1f}MB). Please keep under 2GB for optimal performance."
            
            # Basic validation passed
            warnings = []
            if file_size_mb > 500:
                warnings.append(f"Large file ({file_size_mb:.1f}MB) - processing may take significant time")
            elif file_size_mb > 200:
                warnings.append(f"Medium file ({file_size_mb:.1f}MB) - processing may be slower")
            
            if video_path.suffix.lower() != '.mp4':
                warnings.append("MP4 format recommended for best compatibility")
            
            message = "✅ Video file validated"
            if warnings:
                message += f"\n⚠️ Notes: {'; '.join(warnings)}"
            
            return True, message
            
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    def colorize_video(
        self,
        input_video,
        colorization_method: str = "stable",
        render_factor: int = 21,
        enable_temporal_consistency: bool = True,
        enable_edge_enhancement: bool = True,
        enable_color_stabilization: bool = True,
        post_processing_preset: str = "balanced",
        custom_saturation: float = 1.1,
        custom_contrast: float = 1.05,
        temperature_adjustment: float = 0.0,
        tint_adjustment: float = 0.0,
        frame_skip: int = 1,
        progress=gr.Progress()
    ):
        """
        Colorize video with enhanced processing options.
        
        Args:
            input_video: Input video file
            colorization_method: Method to use (stable/artistic)
            render_factor: Quality factor for rendering
            enable_temporal_consistency: Enable temporal consistency
            enable_edge_enhancement: Enable edge enhancement
            enable_color_stabilization: Enable color stabilization
            post_processing_preset: Post-processing preset
            custom_saturation: Custom saturation factor
            custom_contrast: Custom contrast factor
            temperature_adjustment: Color temperature adjustment
            tint_adjustment: Tint adjustment
            frame_skip: Process every Nth frame
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (output_video_path, processing_info)
        """
        if not DEOLDIFY_AVAILABLE:
            return None, """
## ❌ DeOldify Not Available

DeOldify components are not properly installed. Please check your installation.

### Troubleshooting:
1. Ensure all required packages are installed
2. Check that the deoldify module is available
3. Verify your Python environment setup
"""
        
        if input_video is None:
            return None, """
## ⚠️ No Video File

Please upload a video file to colorize.

### Supported formats:
- **MP4 (.mp4)** - Recommended
- **AVI (.avi)** 
- **MOV (.mov)**
- **MKV (.mkv)**

### Upload Tips:
- Keep file size under 200MB for best performance
- Use 720p or 1080p resolution
- Start with short videos (under 30 seconds) for testing
"""
        
        # Validate video file
        is_valid, validation_message = self.validate_video_file(input_video)
        if not is_valid:
            return None, f"""
## ❌ Video File Issue

{validation_message}

### What to try:
1. **Check format**: Use MP4, AVI, MOV, or MKV
2. **Reduce file size**: Compress video to under 200MB
3. **Test with a shorter clip**: Try 10-30 seconds first
4. **Use MP4 format**: Best compatibility and performance

### Tools for video conversion:
- **HandBrake** (free, cross-platform)
- **FFmpeg** command line tool
- **Online converters** (CloudConvert, etc.)
"""
        
        if not self.models_available or self.multi_colorizer is None:
            return None, f"""
## ❌ Models Not Available

Required model files are missing or failed to load.

### Model Status:
{self._format_model_status()}

### What to do:
1. Use the **Download Models** button below to automatically download missing models
2. Or manually download models to the `models/` directory
3. Restart the application after downloading

### Required models:
- **ColorizeVideo_gen.pth** - Video colorization (stable method)
- **ColorizeArtistic_gen.pth** - Artistic image colorization  
- **ColorizeStable_gen.pth** - Stable image colorization
"""
        
        try:
            progress(0.1, desc="Preparing video...")
            
            # Copy input video to processing directory
            input_path = Path(input_video)
            temp_input = self.temp_dir / f"input_{input_path.name}"
            shutil.copy2(input_video, temp_input)
            
            # Prepare video folder structure
            video_dir = Path("./video")
            video_dir.mkdir(exist_ok=True)
            (video_dir / "source").mkdir(exist_ok=True)
            (video_dir / "result").mkdir(exist_ok=True)
            
            # Copy to DeOldify video source folder
            source_path = video_dir / "source" / temp_input.name
            shutil.copy2(temp_input, source_path)
            
            progress(0.2, desc="Setting up colorization...")
            
            # Setup custom post-processing if needed
            custom_post_processing = {}
            
            if post_processing_preset == "custom":
                custom_post_processing = {
                    'saturation': custom_saturation,
                    'contrast': custom_contrast,
                    'color_balance': {
                        'temperature': temperature_adjustment,
                        'tint': tint_adjustment
                    }
                }
                
                self.multi_colorizer.colorizers[colorization_method].set_custom_post_processing(
                    custom_post_processing
                )
            else:
                self.multi_colorizer.colorizers[colorization_method].set_post_processing_preset(
                    post_processing_preset
                )
            
            progress(0.3, desc="Starting colorization...")
            
            # Colorize video
            output_path = self.multi_colorizer.colorize_video(
                file_name=temp_input.name,
                method=colorization_method,
                render_factor=render_factor,
                temporal_consistency=enable_temporal_consistency,
                edge_enhancement=enable_edge_enhancement,
                color_stabilization=enable_color_stabilization,
                post_processing_preset=post_processing_preset,
                frame_skip=frame_skip
            )
            
            progress(0.9, desc="Finalizing output...")
            
            # Copy result to output directory
            final_output = self.output_dir / f"colorized_{output_path.name}"
            shutil.copy2(output_path, final_output)
            
            # Generate processing info
            processing_info = self._generate_processing_info(
                colorization_method, render_factor, enable_temporal_consistency,
                enable_edge_enhancement, enable_color_stabilization,
                post_processing_preset, custom_post_processing, frame_skip
            )
            
            progress(1.0, desc="Complete!")
            
            return str(final_output), processing_info
            
        except Exception as e:
            error_msg = f"""
## ❌ Colorization Failed

**Error:** {str(e)}

### Troubleshooting:
1. **Check video format**: Ensure your video is in a supported format (MP4, AVI, MOV, MKV)
2. **File size**: Very large files may cause memory issues
3. **Video codec**: Some video codecs may not be supported
4. **ffmpeg**: Make sure ffmpeg is properly installed
5. **Models**: Verify all model files are present and valid

### Technical details:
```
{traceback.format_exc()}
```

### Suggestions:
- Try with a smaller video file first
- Use a different video format (MP4 recommended)
- Check the console output for more detailed error information
"""
            logger.error(error_msg)
            return None, error_msg
        
        finally:
            # Cleanup temp files
            if 'temp_input' in locals() and temp_input.exists():
                temp_input.unlink()
    
    def _format_model_status(self):
        """Format model status for display."""
        if not hasattr(self, 'models_status') or not self.models_status:
            return "Unable to check model status"
            
        status_lines = []
        for model_name, exists in self.models_status.items():
            icon = "✅" if exists else "❌"
            config = MODEL_CONFIGS.get(model_name, {})
            desc = config.get('description', 'Unknown model')
            status_lines.append(f"{icon} **{model_name}** - {desc}")
            
        return "\n".join(status_lines)
    
    def get_model_status_info(self):
        """Get detailed model status information."""
        if not hasattr(self, 'model_downloader'):
            return "Model downloader not initialized"
            
        info = self.model_downloader.get_models_info()
        missing_models = self.model_downloader.get_missing_models()
        
        status_text = "## Model Status\n\n"
        
        for model_name, model_info in info.items():
            icon = "✅" if model_info['exists'] else "❌"
            status_text += f"{icon} **{model_name}**\n"
            status_text += f"   - {model_info['description']}\n"
            
            if model_info['exists']:
                status_text += f"   - Size: {model_info['actual_size_mb']} MB\n"
            else:
                status_text += f"   - Expected size: {model_info['expected_size_mb']} MB\n"
            status_text += "\n"
            
        if missing_models:
            status_text += f"\n### Missing Models: {len(missing_models)}\n"
            status_text += "Click 'Download Models' to download missing model files automatically.\n"
        else:
            status_text += "\n### ✅ All models are available!\n"
            
        return status_text
    
    def _generate_processing_info(self, method, render_factor, temporal_consistency,
                                 edge_enhancement, color_stabilization, preset,
                                 custom_params, frame_skip):
        """Generate processing information summary."""
        info = f"""
## Processing Summary

**Colorization Method:** {method.title()}
**Render Factor:** {render_factor}
**Frame Skip:** {frame_skip} (processing every {frame_skip} frame{'s' if frame_skip > 1 else ''})

### Enhancement Features:
- **Temporal Consistency:** {'✓ Enabled' if temporal_consistency else '✗ Disabled'}
- **Edge Enhancement:** {'✓ Enabled' if edge_enhancement else '✗ Disabled'}
- **Color Stabilization:** {'✓ Enabled' if color_stabilization else '✗ Disabled'}

### Post-Processing:
**Preset:** {preset.title()}
"""
        
        if custom_params:
            info += "\n**Custom Parameters:**\n"
            for key, value in custom_params.items():
                if isinstance(value, dict):
                    info += f"- **{key.title()}:**\n"
                    for subkey, subvalue in value.items():
                        info += f"  - {subkey}: {subvalue}\n"
                else:
                    info += f"- **{key.title()}:** {value}\n"
        
        return info
    
    def enhance_single_image(self, input_image, enhancement_preset="balanced"):
        """Enhance a single image using post-processing only."""
        if not DEOLDIFY_AVAILABLE or input_image is None:
            return None, "No image provided or DeOldify not available."
        
        try:
            from PIL import Image
            
            # Load image
            if isinstance(input_image, str):
                image = Image.open(input_image).convert('RGB')
            else:
                image = Image.fromarray(input_image).convert('RGB')
            
            # Apply enhancement
            enhanced = self.post_processor.apply_enhancement_preset(image, enhancement_preset)
            
            # Save enhanced image
            output_path = self.output_dir / f"enhanced_image_{enhancement_preset}.jpg"
            enhanced.save(output_path)
            
            return str(output_path), f"Enhanced using '{enhancement_preset}' preset"
            
        except Exception as e:
            return None, f"Enhancement failed: {str(e)}"


def create_interface():
    """Create the Gradio interface."""
    
    colorizer = GradioVideoColorizer()
    
    # Get device information for display
    try:
        from deoldify import device
        device_info = device.get_device_info()
    except:
        device_info = {'current_device': 'Unknown', 'is_gpu': False, 'cuda_available': False, 'cuda_device_count': 0}
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .file-tips {
        background-color: #f8f9fa;
        border-left: 4px solid #17a2b8;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9em;
        color: #495057;
    }
    .model-status {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .progress-info {
        background-color: #e7f3ff;
        border-left: 4px solid #007bff;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    """
    
    with gr.Blocks(css=css, title="Enhanced DeOldify Video Colorizer") as interface:
        
        gr.Markdown("""
        # 🎨 Enhanced DeOldify Video Colorizer
        
        Transform black and white videos into vibrant color with advanced AI and temporal consistency features!
        
        ### ✨ Key Features:
        - **Temporal Consistency**: Reduces flickering between frames
        - **Edge Enhancement**: Minimizes color fringing artifacts  
        - **Color Stabilization**: Maintains consistent colors throughout the video
        - **Multiple Models**: Choose between Stable and Artistic colorization
        - **Advanced Post-Processing**: Professional-grade enhancement options
        """)
        
        with gr.Tab("🔧 Model Management"):
            gr.Markdown("### Download and manage required model files")
            
            with gr.Row():
                with gr.Column(scale=2):
                    model_status = gr.Markdown(
                        value=colorizer.get_model_status_info(),
                        label="Model Status"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Actions")
                    
                    check_models_btn = gr.Button("🔍 Check Model Status", variant="secondary")
                    download_models_btn = gr.Button("📥 Download Missing Models", variant="primary")
                    
                    gr.Markdown("""
                    ### Instructions:
                    1. **Check Status**: See which models are available
                    2. **Download Models**: Automatically download missing model files
                    3. Models will be saved to the `models/` directory
                    4. Large files (~123MB each) - ensure good internet connection
                    
                    ### Required Models:
                    - `ColorizeVideo_gen.pth` - Video colorization
                    - `ColorizeArtistic_gen.pth` - Artistic image colorization  
                    - `ColorizeStable_gen.pth` - Stable image colorization
                    """)
            
            download_progress = gr.Markdown(visible=False)
        
        with gr.Tab("Video Colorization"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Settings")
                    
                    input_video = gr.File(
                        label="Upload Video (Max: 2GB supported)",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                        type="filepath"
                    )
                    
                    gr.Markdown("""
                    **📁 File Upload Tips:**
                    - **Recommended**: MP4 format for best compatibility
                    - **Size limit**: Up to 2GB supported (larger files take longer to process)
                    - **Resolution**: 720p, 1080p, or even 4K supported
                    - **Duration**: Start with videos under 30 seconds for testing, longer videos supported
                    - **Performance**: Files under 500MB process faster
                    """, elem_classes=["file-tips"])
                    
                    colorization_method = gr.Dropdown(
                        choices=["video", "stable", "artistic"],
                        value="video",
                        label="Colorization Method",
                        info="Video: Specialized for video sequences, Stable: More realistic colors, Artistic: More vibrant/creative"
                    )
                    
                    render_factor = gr.Slider(
                        minimum=10, maximum=50, value=21, step=1,
                        label="Render Quality Factor",
                        info="Higher = better quality but slower (10-50)"
                    )
                    
                    frame_skip = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Frame Skip",
                        info="Process every Nth frame (1=all frames, 2=every other frame, etc.)"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Enhancement Options")
                    
                    enable_temporal_consistency = gr.Checkbox(
                        value=True,
                        label="Temporal Consistency",
                        info="Reduces flickering between frames"
                    )
                    
                    enable_edge_enhancement = gr.Checkbox(
                        value=True,
                        label="Edge Enhancement",
                        info="Reduces color fringing around edges"
                    )
                    
                    enable_color_stabilization = gr.Checkbox(
                        value=True,
                        label="Color Stabilization",
                        info="Maintains consistent colors throughout video"
                    )
                    
                    post_processing_preset = gr.Dropdown(
                        choices=["balanced", "vivid", "soft", "sharp", "cinematic", "custom"],
                        value="balanced",
                        label="Post-Processing Preset",
                        info="Choose enhancement style"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Custom Post-Processing")
                    gr.Markdown("*Only applied when 'custom' preset is selected*")
                    
                    custom_saturation = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.1, step=0.1,
                        label="Saturation",
                        info="Color intensity (1.0 = no change)"
                    )
                    
                    custom_contrast = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.05, step=0.05,
                        label="Contrast",
                        info="Contrast adjustment (1.0 = no change)"
                    )
                    
                    temperature_adjustment = gr.Slider(
                        minimum=-1.0, maximum=1.0, value=0.0, step=0.1,
                        label="Color Temperature",
                        info="Warm/Cool adjustment (-1=cool, +1=warm)"
                    )
                    
                    tint_adjustment = gr.Slider(
                        minimum=-1.0, maximum=1.0, value=0.0, step=0.1,
                        label="Tint",
                        info="Green/Magenta adjustment (-1=green, +1=magenta)"
                    )
            
            with gr.Row():
                colorize_btn = gr.Button("🎨 Colorize Video", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=2):
                    output_video = gr.Video(label="Colorized Video")
                
                with gr.Column(scale=1):
                    processing_info = gr.Markdown(label="Processing Information")
        
        with gr.Tab("Image Enhancement"):
            gr.Markdown("### Enhance Individual Images")
            gr.Markdown("Test post-processing effects on individual images.")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Image", type="filepath")
                    enhancement_preset = gr.Dropdown(
                        choices=["balanced", "vivid", "soft", "sharp", "cinematic"],
                        value="balanced",
                        label="Enhancement Preset"
                    )
                    enhance_btn = gr.Button("✨ Enhance Image", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(label="Enhanced Image")
                    image_info = gr.Markdown()
        
        with gr.Tab("About & Tips"):
            gr.Markdown("""
            ## 📖 About Enhanced DeOldify
            
            This enhanced version of DeOldify includes several improvements for video colorization:
            
            ### 🔧 Technical Improvements:
            
            **Temporal Consistency**
            - Uses optical flow to track motion between frames
            - Reduces color flickering and maintains consistency
            - Adaptive blending based on motion magnitude
            
            **Edge Enhancement**
            - Detects edges in original grayscale footage
            - Preserves luminance information at edges
            - Reduces color bleeding and fringing artifacts
            
            **Color Stabilization**
            - Tracks dominant color palettes across frames
            - Applies palette consistency to reduce color shifts
            - Maintains visual coherence throughout the video
            
            ### 💡 Usage Tips:
            
            1. **Render Factor**: Start with 21 for good quality/speed balance
            2. **Frame Skip**: Use 2-3 for faster processing on long videos
            3. **Method Selection**: 
               - Use "Stable" for realistic historical footage
               - Use "Artistic" for more creative/stylized results
            4. **Post-Processing**: 
               - "Balanced" works well for most content
               - "Cinematic" adds film-like color grading
               - "Vivid" for more saturated, vibrant colors
            
            ### ⚙️ Current Settings:
            - **Device**: {device_info.get('current_device', 'Unknown')} ({'GPU' if device_info.get('is_gpu', False) else 'CPU'})
            - **CUDA Available**: {device_info.get('cuda_available', False)}
            - **GPU Count**: {device_info.get('cuda_device_count', 0)}
            {f"- **GPU Name**: {device_info.get('cuda_device_name', 'N/A')}" if device_info.get('cuda_available') else ""}
            - **Models**: Stable and Artistic colorization models
            - **Output Format**: MP4 with H.264 encoding
            - **Enhanced Features**: Temporal consistency, edge enhancement, color stabilization
            
            ### 📝 Research References:
            This implementation incorporates techniques from recent research papers:
            - Multimodal semantic-aware colorization
            - Unified diffusion frameworks for video colorization  
            - Palette-guided consistency methods
            """)
        
        # Event handlers
        check_models_btn.click(
            fn=colorizer.get_model_status_info,
            outputs=[model_status]
        )
        
        download_models_btn.click(
            fn=colorizer.download_missing_models,
            outputs=[download_progress, model_status]
        )
        
        colorize_btn.click(
            fn=colorizer.colorize_video,
            inputs=[
                input_video, colorization_method, render_factor,
                enable_temporal_consistency, enable_edge_enhancement, 
                enable_color_stabilization, post_processing_preset,
                custom_saturation, custom_contrast, temperature_adjustment,
                tint_adjustment, frame_skip
            ],
            outputs=[output_video, processing_info]
        )
        
        enhance_btn.click(
            fn=colorizer.enhance_single_image,
            inputs=[input_image, enhancement_preset],
            outputs=[output_image, image_info]
        )
    
    return interface


if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    
    # Launch with appropriate settings for local use
    interface.launch(
        share=False,  # Run locally by default - no public link
        inbrowser=True,  # Open in browser automatically
        server_name="127.0.0.1",  # Local connections only
        server_port=7860  # Default Gradio port
    )