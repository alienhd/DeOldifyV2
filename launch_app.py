#!/usr/bin/env python3
"""
Startup script for Enhanced DeOldify Video Colorization
"""
import sys
import os
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    # Mapping of package names to their actual import names
    package_imports = {
        "gradio": "gradio",
        "opencv-python": "cv2",
        "pillow": "PIL",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "ffmpeg-python": "ffmpeg",
        "yt-dlp": "yt_dlp",
    }

    missing_packages = []
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install missing packages with:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffmpeg not found. Video processing may not work properly.")
        logger.info("Please install ffmpeg:")
        logger.info("Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        logger.info("MacOS: brew install ffmpeg")
        logger.info("Windows: Download from https://ffmpeg.org/download.html")
        return False


def setup_directories():
    """Setup required directories."""
    directories = [
        "video/source",
        "video/result",
        "video/bwframes",
        "video/colorframes",
        "video/audio",
        "gradio_outputs",
        "temp",
        "result_images",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def launch_gradio_app():
    """Launch the Gradio application."""
    try:
        logger.info("Starting Enhanced DeOldify Gradio Application...")

        # Import and run the Gradio app
        from gradio_app import create_interface

        interface = create_interface()

        logger.info("Gradio interface created successfully!")
        logger.info("Launching application...")

        interface.launch(
            share=True, inbrowser=True, server_name="0.0.0.0", server_port=7860
        )

    except ImportError as e:
        logger.error(f"Failed to import Gradio app: {e}")
        logger.info("Make sure all dependencies are installed correctly.")
        return False
    except Exception as e:
        logger.error(f"Failed to launch Gradio app: {e}")
        return False

    return True


def main():
    """Main startup function."""
    logger.info("=== Enhanced DeOldify Video Colorizer ===")
    logger.info("Initializing application...")

    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    # Check ffmpeg
    logger.info("Checking ffmpeg...")
    check_ffmpeg()  # Warning only, not critical

    # Setup directories
    logger.info("Setting up directories...")
    setup_directories()

    # Launch application
    logger.info("All checks passed. Launching application...")
    if not launch_gradio_app():
        sys.exit(1)


if __name__ == "__main__":
    main()
