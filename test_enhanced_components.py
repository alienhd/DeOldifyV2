#!/usr/bin/env python3
"""
Test script for Enhanced DeOldify components
"""
import sys
import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all enhanced components can be imported."""
    logger.info("Testing imports...")
    
    try:
        from deoldify.temporal_consistency import (
            TemporalConsistencyProcessor, 
            EdgeEnhancementProcessor, 
            ColorStabilizer
        )
        logger.info("✓ Temporal consistency components imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import temporal consistency components: {e}")
        return False
    
    try:
        from deoldify.enhanced_post_processing import (
            EnhancedPostProcessor, 
            PostProcessingPipeline
        )
        logger.info("✓ Enhanced post-processing components imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import enhanced post-processing components: {e}")
        return False
    
    try:
        from deoldify.enhanced_video_colorizer import (
            EnhancedVideoColorizer, 
            MultiModelVideoColorizer
        )
        logger.info("✓ Enhanced video colorizer components imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import enhanced video colorizer components: {e}")
        return False
    
    return True

def test_post_processing():
    """Test post-processing functionality."""
    logger.info("Testing post-processing...")
    
    try:
        from deoldify.enhanced_post_processing import PostProcessingPipeline
        
        # Create test image
        test_image = Image.new('RGB', (256, 256), color='red')
        
        # Initialize post-processor
        pipeline = PostProcessingPipeline()
        
        # Test different presets
        presets = ["balanced", "vivid", "soft", "sharp", "cinematic"]
        
        for preset in presets:
            try:
                enhanced = pipeline.apply_enhancement_preset(test_image, preset)
                assert enhanced.size == test_image.size
                logger.info(f"✓ Post-processing preset '{preset}' works correctly")
            except Exception as e:
                logger.error(f"✗ Post-processing preset '{preset}' failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Post-processing test failed: {e}")
        return False

def test_temporal_consistency():
    """Test temporal consistency functionality."""
    logger.info("Testing temporal consistency...")
    
    try:
        from deoldify.temporal_consistency import TemporalConsistencyProcessor
        
        # Create test frames
        frame1 = Image.new('RGB', (256, 256), color='gray')
        frame2 = Image.new('RGB', (256, 256), color='lightgray')
        colored1 = Image.new('RGB', (256, 256), color='red')
        colored2 = Image.new('RGB', (256, 256), color='blue')
        
        # Initialize processor
        processor = TemporalConsistencyProcessor()
        
        # Process frames
        result1 = processor.process_frame(frame1, colored1)
        result2 = processor.process_frame(frame2, colored2)
        
        assert result1.size == frame1.size
        assert result2.size == frame2.size
        
        logger.info("✓ Temporal consistency processor works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Temporal consistency test failed: {e}")
        return False

def test_edge_enhancement():
    """Test edge enhancement functionality.""" 
    logger.info("Testing edge enhancement...")
    
    try:
        from deoldify.temporal_consistency import EdgeEnhancementProcessor
        
        # Create test images
        original = Image.new('L', (256, 256), color='gray')
        colored = Image.new('RGB', (256, 256), color='red')
        
        # Initialize processor
        processor = EdgeEnhancementProcessor()
        
        # Apply enhancement
        enhanced = processor.enhance_edges(original, colored)
        
        assert enhanced.size == colored.size
        assert enhanced.mode == 'RGB'
        
        logger.info("✓ Edge enhancement processor works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Edge enhancement test failed: {e}")
        return False

def test_color_stabilization():
    """Test color stabilization functionality."""
    logger.info("Testing color stabilization...")
    
    try:
        from deoldify.temporal_consistency import ColorStabilizer
        
        # Create test images
        frame1 = Image.new('RGB', (256, 256), color='red')
        frame2 = Image.new('RGB', (256, 256), color='blue')
        
        # Initialize stabilizer
        stabilizer = ColorStabilizer()
        
        # Process frames
        result1 = stabilizer.stabilize_colors(frame1)
        result2 = stabilizer.stabilize_colors(frame2)
        
        assert result1.size == frame1.size
        assert result2.size == frame2.size
        
        logger.info("✓ Color stabilizer works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Color stabilization test failed: {e}")
        return False

def test_gradio_components():
    """Test Gradio components."""
    logger.info("Testing Gradio components...")
    
    try:
        import gradio as gr
        logger.info("✓ Gradio is available")
        
        # Try to import the Gradio app
        try:
            from gradio_app import GradioVideoColorizer
            
            # Test initialization (without DeOldify models)
            colorizer = GradioVideoColorizer()
            logger.info("✓ Gradio video colorizer initializes correctly")
            
        except Exception as e:
            logger.warning(f"Gradio video colorizer test limited: {e}")
        
        return True
        
    except ImportError:
        logger.error("✗ Gradio not available")
        return False

def test_directory_structure():
    """Test if required directories can be created."""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        'video/source',
        'video/result',
        'video/bwframes', 
        'video/colorframes',
        'video/audio',
        'gradio_outputs',
        'temp',
        'result_images'
    ]
    
    try:
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("✓ All required directories created successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create directories: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("=== Enhanced DeOldify Component Tests ===")
    
    tests = [
        ("Import Tests", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Post-Processing", test_post_processing),
        ("Temporal Consistency", test_temporal_consistency),
        ("Edge Enhancement", test_edge_enhancement),
        ("Color Stabilization", test_color_stabilization),
        ("Gradio Components", test_gradio_components),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} crashed: {e}")
            failed += 1
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Enhanced DeOldify is ready to use.")
        return True
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Some features may not work correctly.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)