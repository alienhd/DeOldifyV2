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

# Import DeOldify components
try:
    from deoldify.enhanced_video_colorizer import EnhancedVideoColorizer, MultiModelVideoColorizer
    from deoldify.visualize import get_stable_video_colorizer, get_artistic_video_colorizer
    from deoldify.enhanced_post_processing import PostProcessingPipeline
    from deoldify import device
    from deoldify.device_id import DeviceId
    
    # Set device (CPU for safety, can be changed to GPU if available)
    try:
        device.set(device=DeviceId.GPU0)
        device_info = "GPU0"
    except:
        device.set(device=DeviceId.CPU)
        device_info = "CPU"
    
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
        
        if DEOLDIFY_AVAILABLE:
            self.setup_colorizers()
        
        self.post_processor = PostProcessingPipeline() if DEOLDIFY_AVAILABLE else None
        
    def setup_colorizers(self):
        """Setup different colorization models."""
        try:
            # Initialize colorizers
            stable_vis = get_stable_video_colorizer(render_factor=21)
            artistic_vis = get_artistic_video_colorizer(render_factor=35)
            
            # Create enhanced colorizers
            self.multi_colorizer = MultiModelVideoColorizer()
            
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
            
            self.multi_colorizer.add_colorizer("stable", stable_enhanced)
            self.multi_colorizer.add_colorizer("artistic", artistic_enhanced)
            
            logger.info("Colorizers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup colorizers: {e}")
            self.multi_colorizer = None
    
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
            return None, "DeOldify components not available. Please check installation."
        
        if input_video is None:
            return None, "Please upload a video file."
        
        if self.multi_colorizer is None:
            return None, "Colorizers not properly initialized."
        
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
            error_msg = f"Colorization failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            return None, error_msg
        
        finally:
            # Cleanup temp files
            if temp_input.exists():
                temp_input.unlink()
    
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
        
        with gr.Tab("Video Colorization"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Settings")
                    
                    input_video = gr.File(
                        label="Upload Video",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                        type="filepath"
                    )
                    
                    colorization_method = gr.Dropdown(
                        choices=["stable", "artistic"],
                        value="stable",
                        label="Colorization Method",
                        info="Stable: More realistic colors, Artistic: More vibrant/creative"
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
            - **Device**: CPU (change to GPU in code for faster processing)
            - **Models**: Stable and Artistic colorization models
            - **Output Format**: MP4 with H.264 encoding
            
            ### 📝 Research References:
            This implementation incorporates techniques from recent research papers:
            - Multimodal semantic-aware colorization
            - Unified diffusion frameworks for video colorization  
            - Palette-guided consistency methods
            """)
        
        # Event handlers
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
    
    # Launch with appropriate settings
    interface.launch(
        share=True,  # Create public link
        inbrowser=True,  # Open in browser automatically
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860  # Default Gradio port
    )