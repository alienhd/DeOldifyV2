# Enhanced DeOldify V2: Advanced Video Colorization 🎨

A significantly enhanced version of DeOldify with advanced video colorization capabilities, temporal consistency, reduced flickering, and a comprehensive Gradio UI.

## 🆕 New Features & Enhancements

### 🎬 Advanced Video Processing
- **Temporal Consistency**: Reduces flickering between frames using optical flow
- **Edge Enhancement**: Minimizes color fringing around edges 
- **Color Stabilization**: Maintains consistent color palettes throughout videos
- **Frame Interpolation**: Smart frame skipping with interpolation for faster processing

### 🎨 Enhanced Post-Processing
- **Multiple Enhancement Presets**: Balanced, Vivid, Soft, Sharp, Cinematic
- **Advanced Filtering**: Bilateral filtering, guided filtering, unsharp masking
- **Color Grading**: Temperature/tint adjustments, selective color correction
- **Adaptive Enhancement**: CLAHE, selective color adjustments

### 🖥️ Comprehensive Gradio UI
- **Intuitive Interface**: User-friendly web interface for all features
- **Real-time Previews**: Instant feedback on parameter changes
- **Batch Processing**: Process multiple videos with consistent settings
- **Progress Tracking**: Real-time progress updates during processing

### 📚 Research-Based Improvements
Based on the latest research papers included in this repository:
- Multimodal semantic-aware automatic colorization (2404.16678v1.pdf)
- VanGogh: Unified multimodal diffusion framework (2501.09499v1.pdf)  
- Consistent video colorization via palette guidance (2501.19331v1.pdf)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/alienhd/DeOldifyV2.git
cd DeOldifyV2

# Install dependencies
pip install gradio opencv-python pillow numpy matplotlib ffmpeg-python yt-dlp

# Install FFmpeg (if not already installed)
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### 2. Launch the Application

```bash
# Method 1: Use the launch script
python launch_app.py

# Method 2: Run Gradio app directly  
python gradio_app.py

# Method 3: Test components first
python test_enhanced_components.py
```

### 3. Access the Web Interface

Once launched, the application will be available at:
- Local: `http://localhost:7860`
- Public URL: Displayed in terminal (if sharing enabled)

## 📖 Usage Guide

### Video Colorization

1. **Upload Video**: Drag and drop your black & white video file
2. **Choose Method**: 
   - **Stable**: More realistic, conservative colorization
   - **Artistic**: More vibrant, creative colorization
3. **Adjust Settings**:
   - **Render Factor**: 10-50 (higher = better quality, slower)
   - **Frame Skip**: 1-5 (higher = faster processing)
4. **Enable Enhancements**:
   - ✅ **Temporal Consistency**: Reduces flickering
   - ✅ **Edge Enhancement**: Reduces color bleeding
   - ✅ **Color Stabilization**: Consistent colors
5. **Select Post-Processing**:
   - **Balanced**: Good for most videos
   - **Vivid**: More saturated colors
   - **Cinematic**: Film-like color grading
   - **Custom**: Fine-tune individual parameters

### Image Enhancement

Test post-processing effects on individual images:
1. Upload an image
2. Select enhancement preset
3. Preview results instantly

## 🔧 Technical Details

### Temporal Consistency Algorithm
```python
# Uses optical flow for frame-to-frame consistency
flow = optical_flow.calc(prev_frame, current_frame)
warped_prev = cv2.remap(prev_colored, flow_map)
consistent_frame = adaptive_blend(current_colored, warped_prev)
```

### Edge Enhancement Process
```python
# Preserves luminance at detected edges
edges = cv2.Canny(original_gray)
enhanced = blend_with_original_at_edges(colored, original, edges)
```

### Color Stabilization Method
```python
# Tracks dominant color palettes
palette = extract_palette(frame)
stabilized = apply_palette_consistency(frame, reference_palette)
```

## 📊 Performance Comparison

| Feature | Original DeOldify | Enhanced DeOldify V2 |
|---------|------------------|---------------------|
| Temporal Consistency | ❌ | ✅ Advanced optical flow |
| Edge Artifacts | 🟡 Some fringing | ✅ Significantly reduced |
| Color Stability | 🟡 Variable | ✅ Palette-guided consistency |
| Post-Processing | 🟡 Basic | ✅ Professional-grade |
| User Interface | ❌ Jupyter only | ✅ Comprehensive Gradio UI |
| Batch Processing | ❌ | ✅ Multiple videos |
| Real-time Preview | ❌ | ✅ Instant feedback |

## 🎯 Recommended Settings

### For Historical Footage
- **Method**: Stable
- **Render Factor**: 25-35
- **Preset**: Balanced or Cinematic
- **All enhancements**: Enabled

### For Animation/Cartoons
- **Method**: Artistic  
- **Render Factor**: 35-45
- **Preset**: Vivid
- **Color Stabilization**: Essential

### For Fast Processing
- **Render Factor**: 15-20
- **Frame Skip**: 2-3
- **Preset**: Balanced
- **Temporal Consistency**: Still recommended

## 🛠️ Advanced Configuration

### Custom Post-Processing
```python
# Example custom parameters
custom_params = {
    'saturation': 1.2,
    'contrast': 1.1,
    'color_balance': {
        'temperature': 0.1,  # Slightly warm
        'tint': -0.05       # Slightly green
    },
    'unsharp_mask': {
        'radius': 1.5,
        'amount': 1.0
    }
}
```

### Temporal Consistency Tuning
```python
processor = TemporalConsistencyProcessor(
    alpha=0.7,              # Consistency strength
    flow_threshold=0.5      # Motion sensitivity
)
```

## 📁 Project Structure

```
DeOldifyV2/
├── deoldify/
│   ├── temporal_consistency.py      # Temporal processing
│   ├── enhanced_post_processing.py  # Advanced filters
│   ├── enhanced_video_colorizer.py  # Main colorizer
│   └── visualize.py                 # Updated core module
├── gradio_app.py                    # Web interface
├── launch_app.py                    # Startup script
├── test_enhanced_components.py      # Testing suite
├── *.pdf                           # Research papers
└── README_ENHANCED.md              # This file
```

## 🔬 Research Integration

The enhancements are based on three key research papers:

1. **Multimodal Semantic-Aware Colorization** (2404.16678v1.pdf)
   - Improved semantic understanding for better color choices
   - Diffusion-based approaches for more realistic results

2. **VanGogh Framework** (2501.09499v1.pdf)  
   - Unified multimodal approach supporting text prompts
   - Advanced temporal consistency methods

3. **Palette-Guided Consistency** (2501.19331v1.pdf)
   - Color palette tracking across frames
   - Consistency-preserving colorization pipeline

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Additional Models**: Integration of newer colorization models
- **GPU Optimization**: Better GPU memory management
- **Mobile Support**: Responsive UI for mobile devices
- **API Endpoints**: REST API for programmatic access
- **Cloud Integration**: Support for cloud storage/processing

## 📄 License

This project maintains the same license as the original DeOldify project.

## 🙏 Acknowledgments

- Original DeOldify team for the foundational work
- Research paper authors for algorithmic innovations
- OpenCV and PIL communities for image processing tools
- Gradio team for the excellent UI framework

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'deoldify'"**
```bash
# Make sure you're in the DeOldifyV2 directory
cd DeOldifyV2
python gradio_app.py
```

**"FFmpeg not found"**
```bash
# Install FFmpeg for your system
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

**"CUDA out of memory"**
- Reduce render_factor (try 15-20)
- Enable frame_skip (try 2-3)
- Use CPU instead of GPU in device settings

**Slow processing**
- Use GPU if available (edit device settings in code)
- Increase frame_skip for faster processing
- Reduce render_factor for speed vs quality trade-off

### Performance Tips

1. **GPU Setup**: Edit `gradio_app.py` to use GPU:
   ```python
   device.set(device=DeviceId.GPU0)  # Instead of CPU
   ```

2. **Memory Management**: For large videos:
   - Process in segments
   - Use frame skipping
   - Lower render factor

3. **Quality vs Speed**: 
   - High quality: render_factor=35+, frame_skip=1
   - Balanced: render_factor=21, frame_skip=1-2  
   - Fast: render_factor=15, frame_skip=2-3

## 📞 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Run `python test_enhanced_components.py` to diagnose issues
3. Check the original DeOldify documentation
4. Open an issue on GitHub with detailed information

---

**Happy Colorizing! 🎨✨**