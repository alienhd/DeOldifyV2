#!/usr/bin/env python3
"""
Model downloader utility for DeOldify
Downloads required model files if they don't exist
"""
import os
import requests
import logging
from pathlib import Path
from typing import Dict, Optional
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs and expected file hashes
MODEL_CONFIGS = {
    'ColorizeVideo_gen.pth': {
        'url': 'https://data.deepai.org/deoldify/ColorizeVideo_gen.pth',
        'description': 'Video Colorization Model (Stable)',
        'size_mb': 123  # Approximate size
    },
    'ColorizeArtistic_gen.pth': {
        'url': 'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth', 
        'description': 'Image Colorization Model (Artistic)',
        'size_mb': 123
    },
    'ColorizeStable_gen.pth': {
        'url': 'https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeStable_gen.pth?dl=1',
        'description': 'Image Colorization Model (Stable)', 
        'size_mb': 123,
        'alt_urls': [
            'https://data.deepai.org/deoldify/ColorizeStable_gen.pth',
            'https://github.com/jantic/DeOldify/releases/download/v1.0/ColorizeStable_gen.pth'
        ]
    }
}


class ModelDownloader:
    """Handles downloading and managing DeOldify model files."""
    
    def __init__(self, models_dir: str = "./models"):
        """Initialize model downloader."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model file exists."""
        model_path = self.models_dir / model_name
        return model_path.exists() and model_path.stat().st_size > 1000  # At least 1KB
        
    def get_missing_models(self) -> list:
        """Get list of missing required models."""
        missing = []
        for model_name in MODEL_CONFIGS.keys():
            if not self.check_model_exists(model_name):
                missing.append(model_name)
        return missing
        
    def download_model(self, model_name: str, progress_callback=None) -> bool:
        """
        Download a specific model file.
        
        Args:
            model_name: Name of the model file to download
            progress_callback: Optional callback function for progress updates
            
        Returns:
            True if download successful, False otherwise
        """
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        config = MODEL_CONFIGS[model_name]
        model_path = self.models_dir / model_name
        
        # Check if already exists
        if self.check_model_exists(model_name):
            logger.info(f"Model {model_name} already exists")
            return True
            
        logger.info(f"Downloading {config['description']} ({config['size_mb']}MB)")
        
        # Try main URL first, then alternative URLs if available
        urls_to_try = [config['url']]
        if 'alt_urls' in config:
            urls_to_try.extend(config['alt_urls'])
            
        for url in urls_to_try:
            logger.info(f"Trying URL: {url}")
            
            try:
                # Download with progress
                response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress = downloaded / total_size
                                progress_callback(progress, f"Downloading {model_name}")
                                
                logger.info(f"Successfully downloaded {model_name}")
                return True
                
            except requests.RequestException as e:
                logger.warning(f"Failed to download from {url}: {e}")
                # Clean up partial file
                if model_path.exists():
                    model_path.unlink()
                continue
            except Exception as e:
                logger.warning(f"Unexpected error downloading from {url}: {e}")
                if model_path.exists():
                    model_path.unlink()
                continue
                
        logger.error(f"Failed to download {model_name} from all available URLs")
        return False
            
    def download_all_models(self, progress_callback=None) -> Dict[str, bool]:
        """
        Download all missing models.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping model names to download success status
        """
        missing_models = self.get_missing_models()
        results = {}
        
        if not missing_models:
            logger.info("All models are already present")
            return {}
            
        logger.info(f"Downloading {len(missing_models)} missing models...")
        
        for i, model_name in enumerate(missing_models):
            if progress_callback:
                overall_progress = i / len(missing_models)
                progress_callback(overall_progress, f"Downloading models ({i+1}/{len(missing_models)})")
                
            success = self.download_model(model_name, progress_callback)
            results[model_name] = success
            
        return results
        
    def verify_models(self) -> Dict[str, bool]:
        """Verify that all required models are present and valid."""
        verification = {}
        
        for model_name in MODEL_CONFIGS.keys():
            exists = self.check_model_exists(model_name)
            verification[model_name] = exists
            
            if exists:
                logger.info(f"✓ {model_name} - OK")
            else:
                logger.warning(f"✗ {model_name} - Missing")
                
        return verification
        
    def get_models_info(self) -> Dict[str, Dict]:
        """Get information about all models."""
        info = {}
        
        for model_name, config in MODEL_CONFIGS.items():
            model_path = self.models_dir / model_name
            exists = self.check_model_exists(model_name)
            
            info[model_name] = {
                'description': config['description'],
                'expected_size_mb': config['size_mb'],
                'exists': exists,
                'path': str(model_path),
                'actual_size_mb': round(model_path.stat().st_size / (1024*1024), 1) if exists else 0
            }
            
        return info


def main():
    """CLI interface for model downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DeOldify model files")
    parser.add_argument('--models-dir', default='./models', help='Directory to store models')
    parser.add_argument('--check', action='store_true', help='Check model status without downloading')
    parser.add_argument('--download', nargs='*', help='Download specific models (or all if no names given)')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.check:
        print("\nModel Status:")
        verification = downloader.verify_models()
        missing = [name for name, exists in verification.items() if not exists]
        
        if missing:
            print(f"\nMissing models: {missing}")
            print("Run with --download to download missing models")
        else:
            print("\nAll models are present!")
            
    elif args.download is not None:
        if args.download:  # Specific models requested
            for model_name in args.download:
                success = downloader.download_model(model_name)
                if success:
                    print(f"✓ Downloaded {model_name}")
                else:
                    print(f"✗ Failed to download {model_name}")
        else:  # Download all missing
            results = downloader.download_all_models()
            if results:
                success_count = sum(results.values())
                print(f"\nDownloaded {success_count}/{len(results)} models successfully")
            else:
                print("All models already present")
    else:
        # Show status by default
        print("DeOldify Model Downloader")
        print("Use --check to verify models or --download to download missing models")
        
        info = downloader.get_models_info()
        print(f"\nFound {len([i for i in info.values() if i['exists']])}/{len(info)} models")


if __name__ == "__main__":
    main()