# Enhanced DeOldify Setup and Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install gradio requests
```

### 2. Download Models
```bash
# Check which models are missing
python model_downloader.py --check

# Download all missing models (automatic)
python model_downloader.py --download

# Or download specific models
python model_downloader.py --download ColorizeVideo_gen.pth
```

### 3. Launch Application
```bash
# Using the launch script
python launch_app.py

# Or directly
python gradio_app.py
```

The application will start at `http://127.0.0.1:7860` (local access only).

## Key Improvements

### 🔧 Model Management
- **Automatic Model Download**: Missing models are automatically detected and can be downloaded with one click
- **Model Status Checking**: See which models are available and their sizes
- **CLI Tools**: Command-line interface for model management

### 🎬 Video Processing
- **File Validation**: Automatic checking of video format and size
- **Size Limits**: Recommendations for optimal file sizes (under 200MB)
- **Format Support**: MP4, AVI, MOV, MKV with MP4 recommended
- **Progress Tracking**: Real-time progress updates during processing

### 🔒 Local Operation
- **No Public Links**: Runs locally only (127.0.0.1) instead of creating public tunnels
- **Faster Uploads**: No network overhead from Gradio's sharing system
- **Privacy**: All processing happens on your local machine

### ❌ Better Error Handling
- **Clear Error Messages**: Detailed explanations when something goes wrong
- **Troubleshooting Guides**: Step-by-step instructions for common issues
- **Graceful Degradation**: Application works even with missing models

## Model Files

The application requires these model files (automatically downloaded):

| Model File | Purpose | Size | Status |
|------------|---------|------|--------|
| `ColorizeVideo_gen.pth` | Video colorization (stable) | ~123MB | Required |
| `ColorizeArtistic_gen.pth` | Artistic image colorization | ~123MB | Optional |
| `ColorizeStable_gen.pth` | Stable image colorization | ~123MB | Optional |

## Troubleshooting

### Models Not Loading
1. Check model status: `python model_downloader.py --check`
2. Download missing models: `python model_downloader.py --download`
3. Verify models directory contains `.pth` files

### Video Upload Issues
1. **File too large**: Keep videos under 200MB for best performance
2. **Wrong format**: Use MP4 for best compatibility
3. **Slow processing**: Try shorter videos (10-30 seconds) for testing

### Performance Tips
1. **Start small**: Test with short, low-resolution videos first
2. **Use MP4**: Best format for compatibility and speed
3. **Frame skipping**: Use frame_skip=2 or 3 for faster processing
4. **Render factor**: Lower values (15-25) for faster processing

## Command Line Tools

### Model Downloader
```bash
# Check status
python model_downloader.py --check

# Download all missing models
python model_downloader.py --download

# Download specific model
python model_downloader.py --download ColorizeVideo_gen.pth

# Help
python model_downloader.py --help
```

## Configuration Options

### Launch Settings
In `launch_app.py` and `gradio_app.py`:
- `share=False`: No public links (local only)
- `server_name="127.0.0.1"`: Local access only  
- `server_port=7860`: Default port

### Processing Settings
- **Render Factor**: 10-50 (higher = better quality, slower)
- **Frame Skip**: 1-5 (higher = faster, lower quality)
- **Enhancement Features**: Temporal consistency, edge enhancement, color stabilization

## Known Issues

1. **FFmpeg dependency**: Some features require ffmpeg to be installed
2. **Memory usage**: Large videos may require significant RAM
3. **Processing time**: High-quality settings can take considerable time

## Changes Made

This enhanced version addresses the original issues:

1. **Fixed slow video uploads** by running locally instead of using Gradio's public sharing
2. **Resolved missing model errors** with automatic download functionality
3. **Improved user experience** with better error messages and guidance
4. **Added model management** with a dedicated interface tab
5. **Enhanced file validation** to catch issues before processing starts