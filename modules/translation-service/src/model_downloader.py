#!/usr/bin/env python3
"""
Model Downloader for vLLM Translation Server

Handles downloading, caching, and validation of LLM models for translation.
Optimized for Qwen2.5-14B-Instruct-AWQ model.
"""

import os
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Hugging Face imports
try:
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

# Transformers for model validation
try:
    from transformers import AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information and metadata"""
    name: str
    repo_id: str
    size_gb: float
    quantization: str
    description: str
    languages: List[str]
    recommended_gpu_memory: float
    download_url: Optional[str] = None
    local_path: Optional[str] = None
    is_downloaded: bool = False
    checksum: Optional[str] = None

class ModelDownloader:
    """
    Model downloader and manager for vLLM translation server
    
    Features:
    - Automatic model download from Hugging Face
    - Model validation and integrity checking
    - Caching and storage management
    - Progress tracking and resumable downloads
    - Model metadata management
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 hf_token: Optional[str] = None):
        """
        Initialize model downloader
        
        Args:
            cache_dir: Directory for model cache (default: ~/.cache/livetranslate/models)
            hf_token: Hugging Face token for private models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "livetranslate" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) if HF_AVAILABLE else None
        
        # Supported models for translation
        self.supported_models = {
            "qwen2.5-14b-instruct-awq": ModelInfo(
                name="Qwen2.5-14B-Instruct-AWQ",
                repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
                size_gb=8.2,
                quantization="AWQ",
                description="High-quality instruction-tuned model with AWQ quantization",
                languages=["en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar"],
                recommended_gpu_memory=12.0
            ),
            "qwen2.5-7b-instruct-awq": ModelInfo(
                name="Qwen2.5-7B-Instruct-AWQ",
                repo_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
                size_gb=4.8,
                quantization="AWQ",
                description="Smaller instruction-tuned model with AWQ quantization",
                languages=["en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar"],
                recommended_gpu_memory=8.0
            ),
            "qwen2.5-14b-instruct": ModelInfo(
                name="Qwen2.5-14B-Instruct",
                repo_id="Qwen/Qwen2.5-14B-Instruct",
                size_gb=28.0,
                quantization="None",
                description="Full precision instruction-tuned model",
                languages=["en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar"],
                recommended_gpu_memory=32.0
            )
        }
        
        # Default model for translation
        self.default_model = "qwen2.5-14b-instruct-awq"
    
    def list_supported_models(self) -> Dict[str, ModelInfo]:
        """Get list of supported models"""
        return self.supported_models.copy()
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.supported_models.get(model_name.lower())
    
    async def check_model_availability(self, model_name: str) -> Tuple[bool, str]:
        """
        Check if model is available for download
        
        Args:
            model_name: Model identifier
            
        Returns:
            Tuple of (is_available, status_message)
        """
        if not HF_AVAILABLE:
            return False, "Hugging Face Hub not available"
        
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False, f"Model {model_name} not supported"
        
        try:
            # Check if model exists on Hugging Face
            model_files = self.api.list_repo_files(model_info.repo_id)
            
            # Check for required files
            required_files = ["config.json", "tokenizer.json"]
            missing_files = [f for f in required_files if f not in model_files]
            
            if missing_files:
                return False, f"Missing required files: {missing_files}"
            
            return True, "Model available for download"
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                return False, "Model not found on Hugging Face Hub"
            else:
                return False, f"Error checking model: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    async def download_model(self, 
                           model_name: str, 
                           force_download: bool = False,
                           progress_callback: Optional[callable] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Download model to local cache
        
        Args:
            model_name: Model identifier
            force_download: Force re-download even if cached
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (success, message, local_path)
        """
        if not HF_AVAILABLE:
            return False, "Hugging Face Hub not available", None
        
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False, f"Model {model_name} not supported", None
        
        # Check local cache first
        local_path = self.cache_dir / model_name
        if local_path.exists() and not force_download:
            if await self._validate_model(local_path):
                logger.info(f"Model {model_name} already cached at {local_path}")
                return True, "Model already cached", str(local_path)
            else:
                logger.warning(f"Cached model {model_name} is invalid, re-downloading")
        
        try:
            logger.info(f"Downloading model {model_info.name} from {model_info.repo_id}")
            
            # Create progress callback wrapper
            def hf_progress_callback(downloaded: int, total: int):
                if progress_callback:
                    progress_callback(downloaded, total, model_info.name)
            
            # Download model using snapshot_download for full model
            downloaded_path = snapshot_download(
                repo_id=model_info.repo_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                token=self.hf_token,
                resume_download=True
            )
            
            # Validate downloaded model
            if await self._validate_model(local_path):
                # Update model info
                model_info.local_path = str(local_path)
                model_info.is_downloaded = True
                
                # Save model metadata
                await self._save_model_metadata(model_name, model_info)
                
                logger.info(f"Model {model_name} downloaded successfully to {local_path}")
                return True, "Model downloaded successfully", str(local_path)
            else:
                return False, "Downloaded model failed validation", None
                
        except HfHubHTTPError as e:
            error_msg = f"Download failed: HTTP {e.response.status_code}"
            logger.error(error_msg)
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Download failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    
    async def get_local_model_path(self, model_name: str) -> Optional[str]:
        """
        Get local path for a model if it exists and is valid
        
        Args:
            model_name: Model identifier
            
        Returns:
            Local path if model exists, None otherwise
        """
        local_path = self.cache_dir / model_name
        
        if local_path.exists() and await self._validate_model(local_path):
            return str(local_path)
        
        return None
    
    async def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Delete a cached model
        
        Args:
            model_name: Model identifier
            
        Returns:
            Tuple of (success, message)
        """
        local_path = self.cache_dir / model_name
        
        if not local_path.exists():
            return False, f"Model {model_name} not found in cache"
        
        try:
            import shutil
            shutil.rmtree(local_path)
            
            # Remove metadata
            metadata_path = self.cache_dir / f"{model_name}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Model {model_name} deleted from cache")
            return True, "Model deleted successfully"
            
        except Exception as e:
            error_msg = f"Failed to delete model: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    async def list_cached_models(self) -> List[Tuple[str, ModelInfo]]:
        """
        List all cached models with their info
        
        Returns:
            List of (model_name, model_info) tuples
        """
        cached_models = []
        
        for model_name in self.supported_models.keys():
            local_path = self.cache_dir / model_name
            if local_path.exists():
                # Load metadata if available
                metadata_path = self.cache_dir / f"{model_name}.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        model_info = ModelInfo(**metadata)
                        model_info.is_downloaded = True
                        model_info.local_path = str(local_path)
                        
                        cached_models.append((model_name, model_info))
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {model_name}: {e}")
                        # Use default info
                        model_info = self.supported_models[model_name]
                        model_info.is_downloaded = True
                        model_info.local_path = str(local_path)
                        cached_models.append((model_name, model_info))
        
        return cached_models
    
    async def get_cache_size(self) -> float:
        """
        Get total size of model cache in GB
        
        Returns:
            Cache size in GB
        """
        total_size = 0
        
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        
        return total_size / (1024 ** 3)  # Convert to GB
    
    async def _validate_model(self, model_path: Path) -> bool:
        """
        Validate that a model is properly downloaded and usable
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if model is valid
        """
        try:
            # Check for required files
            required_files = ["config.json"]
            for file_name in required_files:
                if not (model_path / file_name).exists():
                    logger.warning(f"Missing required file: {file_name}")
                    return False
            
            # Try to load config with transformers if available
            if TRANSFORMERS_AVAILABLE:
                try:
                    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
                    logger.debug(f"Model config loaded successfully: {config.model_type}")
                except Exception as e:
                    logger.warning(f"Failed to load model config: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    async def _save_model_metadata(self, model_name: str, model_info: ModelInfo):
        """Save model metadata to cache"""
        try:
            metadata_path = self.cache_dir / f"{model_name}.json"
            
            # Convert to dict for JSON serialization
            metadata = {
                "name": model_info.name,
                "repo_id": model_info.repo_id,
                "size_gb": model_info.size_gb,
                "quantization": model_info.quantization,
                "description": model_info.description,
                "languages": model_info.languages,
                "recommended_gpu_memory": model_info.recommended_gpu_memory,
                "local_path": model_info.local_path,
                "is_downloaded": model_info.is_downloaded,
                "download_timestamp": time.time()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save model metadata: {e}")


# CLI interface for testing
async def main():
    """CLI interface for model downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Translation Model Downloader")
    parser.add_argument("--list", action="store_true", help="List supported models")
    parser.add_argument("--download", type=str, help="Download a model")
    parser.add_argument("--check", type=str, help="Check model availability")
    parser.add_argument("--cached", action="store_true", help="List cached models")
    parser.add_argument("--delete", type=str, help="Delete a cached model")
    parser.add_argument("--cache-size", action="store_true", help="Show cache size")
    parser.add_argument("--force", action="store_true", help="Force download even if cached")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)
    
    if args.list:
        print("\nSupported Models:")
        print("-" * 80)
        for name, info in downloader.list_supported_models().items():
            print(f"Name: {name}")
            print(f"  Repository: {info.repo_id}")
            print(f"  Size: {info.size_gb:.1f} GB")
            print(f"  Quantization: {info.quantization}")
            print(f"  Languages: {', '.join(info.languages)}")
            print(f"  Recommended GPU Memory: {info.recommended_gpu_memory:.1f} GB")
            print(f"  Description: {info.description}")
            print()
    
    elif args.check:
        available, message = await downloader.check_model_availability(args.check)
        print(f"Model {args.check}: {'✓ Available' if available else '✗ Not Available'}")
        print(f"Status: {message}")
    
    elif args.download:
        def progress_callback(downloaded, total, model_name):
            if total > 0:
                percent = (downloaded / total) * 100
                print(f"\rDownloading {model_name}: {percent:.1f}% ({downloaded}/{total} bytes)", end="")
        
        print(f"Downloading model: {args.download}")
        success, message, path = await downloader.download_model(
            args.download, 
            force_download=args.force,
            progress_callback=progress_callback
        )
        
        print(f"\n{'✓ Success' if success else '✗ Failed'}: {message}")
        if path:
            print(f"Model path: {path}")
    
    elif args.cached:
        cached = await downloader.list_cached_models()
        if cached:
            print("\nCached Models:")
            print("-" * 80)
            for name, info in cached:
                print(f"Name: {name}")
                print(f"  Path: {info.local_path}")
                print(f"  Size: {info.size_gb:.1f} GB")
                print()
        else:
            print("No cached models found.")
    
    elif args.delete:
        success, message = await downloader.delete_model(args.delete)
        print(f"{'✓ Success' if success else '✗ Failed'}: {message}")
    
    elif args.cache_size:
        size = await downloader.get_cache_size()
        print(f"Total cache size: {size:.2f} GB")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 