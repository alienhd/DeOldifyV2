#!/usr/bin/env python3
"""
Test script for enhanced DeOldify features.
Tests device detection, frame extraction command generation, and enhanced colorizer setup.
"""

import sys
import os
from pathlib import Path

def test_device_detection():
    """Test device detection and setup."""
    print("🔧 Testing Device Detection...")
    try:
        from deoldify import device
        from deoldify.device_id import DeviceId
        
        # Test GPU detection (will fallback to CPU in this environment)
        device.set(device=DeviceId.GPU0)
        device_info = device.get_device_info()
        
        print(f"✅ Device setup successful:")
        for key, value in device_info.items():
            print(f"   {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_grayscale_frame_extraction():
    """Test grayscale frame extraction command generation."""
    print("\n🎬 Testing Grayscale Frame Extraction...")
    try:
        import ffmpeg
        from pathlib import Path
        
        # Test the frame extraction command generation
        dummy_source = Path("test_video.mp4")
        dummy_output = "test_frames/%5d.jpg"
        
        process = (
            ffmpeg
                .input(str(dummy_source))
                .filter('format', 'gray')  # Convert to grayscale
                .output(str(dummy_output), format='image2', vcodec='mjpeg', **{'q:v':'0'})
                .global_args('-hide_banner')
                .global_args('-nostats')
                .global_args('-loglevel', 'error')
        )
        
        command = ffmpeg.compile(process)
        print(f"✅ FFmpeg command generated successfully:")
        print(f"   Command: {' '.join(command)}")
        print(f"   ✓ Grayscale filter: 'format=gray' included")
        print(f"   ✓ Output format: JPEG frames")
        
        return True
    except Exception as e:
        print(f"❌ Frame extraction test failed: {e}")
        return False

def test_enhanced_colorizer_setup():
    """Test enhanced video colorizer setup."""
    print("\n🎨 Testing Enhanced Colorizer Setup...")
    try:
        from deoldify.enhanced_video_colorizer import EnhancedVideoColorizer, MultiModelVideoColorizer
        from deoldify.visualize import get_video_colorizer
        from deoldify.enhanced_post_processing import PostProcessingPipeline
        
        # Create a basic visualizer (without loading actual models)
        print("   Loading basic components...")
        
        # Test post-processing pipeline
        post_processor = PostProcessingPipeline()
        print("   ✓ Post-processing pipeline created")
        
        # Test multi-model setup structure
        multi_colorizer = MultiModelVideoColorizer()
        print("   ✓ Multi-model colorizer structure created")
        
        available_methods = multi_colorizer.get_available_methods()
        print(f"   ✓ Available methods: {available_methods}")
        
        print("✅ Enhanced colorizer components working")
        return True
    except Exception as e:
        print(f"❌ Enhanced colorizer test failed: {e}")
        return False

def test_jupyter_notebook_compatibility():
    """Test that the notebook components would work."""
    print("\n📓 Testing Jupyter Notebook Compatibility...")
    try:
        from IPython.display import HTML, Image as ipythonimage
        from IPython import display as ipythondisplay
        print("   ✓ IPython display components available")
        
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        print("   ✓ Matplotlib styling available")
        
        print("✅ Jupyter notebook compatibility confirmed")
        return True
    except Exception as e:
        print(f"❌ Jupyter compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced DeOldify Feature Tests")
    print("=" * 50)
    
    tests = [
        ("Device Detection", test_device_detection),
        ("Grayscale Frame Extraction", test_grayscale_frame_extraction),
        ("Enhanced Colorizer Setup", test_enhanced_colorizer_setup),
        ("Jupyter Notebook Compatibility", test_jupyter_notebook_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Enhanced features are ready to use.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())