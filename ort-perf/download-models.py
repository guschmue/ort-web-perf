#!/usr/bin/env python3
"""
Download ONNX models from Hugging Face based on ort-models.json configuration.

Usage:
    python download-models.py --category default
    python download-models.py --category default --model resnet50
    python download-models.py --model bert-base-uncased
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    sys.exit(1)


def load_models_config(config_path: str = "ort-models.json") -> List[Dict]:
    """Load the models configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def filter_models(models: List[Dict], category: Optional[str] = None, model_name: Optional[str] = None) -> List[Dict]:
    """Filter models based on category and/or model name."""
    filtered = models
    
    if category:
        filtered = [m for m in filtered if m.get('category') == category]
    
    if model_name:
        filtered = [m for m in filtered if m.get('name') == model_name]
    
    return filtered


def model_exists(model_path: str, base_dir: str = ".") -> bool:
    """Check if the model file already exists."""
    full_path = os.path.join(base_dir, model_path)
    exists = os.path.exists(full_path)
    if exists:
        file_size = os.path.getsize(full_path)
        print(f"  ✓ Model already exists: {full_path} ({file_size:,} bytes)")
    return exists


def extract_file_from_path(model_path: str) -> str:
    """Extract the filename from the model path."""
    return os.path.basename(model_path)


def extract_repo_subpath(model_path: str) -> Optional[str]:
    """
    Extract the subdirectory path within the HF repo.
    For example: 'tjs/resnet-50/onnx/model.onnx' -> 'onnx'
    """
    parts = Path(model_path).parts
    # Find 'onnx' directory in the path
    if 'onnx' in parts:
        idx = parts.index('onnx')
        return '/'.join(parts[idx:])
    return None


def download_model(model: Dict, base_dir: str = ".") -> bool:
    """
    Download a model from Hugging Face if it has an hfrepo tag.
    
    Returns True if download was successful or skipped, False on error.
    """
    name = model.get('name', 'unknown')
    path = model.get('path', '')
    hfrepo = model.get('hfrepo', '').strip().rstrip('/')
    
    if not hfrepo:
        print(f"  ⊘ Skipping '{name}': No hfrepo specified")
        return True
    
    if not path:
        print(f"  ⊘ Skipping '{name}': No path specified")
        return True
    
    # Check if model already exists
    if model_exists(path, base_dir):
        return True
    
    print(f"  ↓ Downloading '{name}' from {hfrepo}")
    
    # Create the target directory
    full_path = os.path.join(base_dir, path)
    target_dir = os.path.dirname(full_path)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Extract the file path within the repo
        repo_subpath = extract_repo_subpath(path)
        filename = extract_file_from_path(path)
        
        # Try to download the specific file
        if repo_subpath:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=hfrepo,
                    filename=repo_subpath,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )
                print(f"    ✓ Downloaded to: {downloaded_path}")
                return True
            except Exception as e:
                print(f"    Warning: Could not download with subpath '{repo_subpath}': {e}")
        
        # Fallback: download entire repo or try without subpath
        print("    Attempting snapshot download...")
        snapshot_path = snapshot_download(
            repo_id=hfrepo,
            allow_patterns=[f"**/{filename}", "**/model*.onnx", "**/*.onnx"],
            local_dir=os.path.join(base_dir, os.path.dirname(path).split('/')[0]),
            local_dir_use_symlinks=False
        )
        print(f"    ✓ Downloaded to: {snapshot_path}")
        return True
        
    except HfHubHTTPError as e:
        print(f"    ✗ HTTP Error downloading '{name}': {e}")
        return False
    except Exception as e:
        print(f"    ✗ Error downloading '{name}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ONNX models from Hugging Face based on ort-models.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models in the 'default' category
  python download-models.py --category default
  
  # Download a specific model
  python download-models.py --model resnet50
  
  # Download a specific model from a category
  python download-models.py --category default --model bert-base-uncased
  
  # Download all models (use with caution - may download many GBs)
  python download-models.py
        """
    )
    
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Category to download (e.g., "default", "local"). If not specified with --model, downloads all categories.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specific model name to download. If not specified, downloads all models in the category.'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='ort-models.json',
        help='Path to the models configuration file (default: ort-models.json)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory for model storage (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    models = load_models_config(args.config)
    print(f"Loaded {len(models)} models from configuration\n")
    
    # Filter models
    filtered_models = filter_models(models, args.category, args.model)
    
    if not filtered_models:
        print("No models match the specified criteria.")
        if args.category or args.model:
            print(f"\nCriteria: category={args.category or 'any'}, model={args.model or 'any'}")
        sys.exit(0)
    
    # Show what will be downloaded
    print(f"Found {len(filtered_models)} model(s) to process:")
    for model in filtered_models:
        hfrepo = model.get('hfrepo', 'N/A')
        print(f"  - {model.get('name')} (category: {model.get('category')}, repo: {hfrepo})")
    print()
    
    # Download models
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, model in enumerate(filtered_models, 1):
        print(f"[{i}/{len(filtered_models)}] Processing: {model.get('name')}")
        
        if not model.get('hfrepo'):
            skip_count += 1
            print(f"  ⊘ Skipping: No hfrepo specified\n")
            continue
        
        if download_model(model, args.base_dir):
            success_count += 1
        else:
            error_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Total models: {len(filtered_models)}")
    print(f"  Already existed/Downloaded: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")
    print("=" * 60)

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
