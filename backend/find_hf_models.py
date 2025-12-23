#!/usr/bin/env python3
"""
Find all Hugging Face models downloaded on your system
Checks default cache locations and calculates sizes
"""

import os
import sys
from pathlib import Path
import platform

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except:
        return 0

def get_folder_size_mb(folder_path):
    """Get total folder size in MB"""
    total_size = 0
    try:
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except:
        pass
    return total_size / (1024 * 1024)

def find_huggingface_cache():
    """Find all Hugging Face cache locations"""
    print("=" * 60)
    print("🔍 SEARCHING FOR HUGGING FACE MODELS")
    print("=" * 60)
    
    # System information
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual Environment: {sys.prefix}")
    else:
        print("Virtual Environment: Not detected (using system Python)")
    
    print("\n" + "=" * 60)
    print("📁 SEARCHING CACHE LOCATIONS")
    print("=" * 60)
    
    cache_locations = []
    total_models = 0
    total_size = 0
    
    # 1. Default Hugging Face cache directory
    default_cache = Path.home() / ".cache" / "huggingface"
    hub_cache = default_cache / "hub"
    
    print(f"\n1. Default HuggingFace Cache:")
    print(f"   Path: {hub_cache}")
    print(f"   Exists: {'✅ YES' if hub_cache.exists() else '❌ NO'}")
    
    if hub_cache.exists():
        models_found = []
        hub_size = 0
        
        for item in hub_cache.iterdir():
            if item.is_dir() and item.name.startswith('models--'):
                # Extract model name
                model_name = item.name.replace('models--', '').replace('--', '/')
                folder_size = get_folder_size_mb(item)
                models_found.append((model_name, folder_size, item))
                hub_size += folder_size
                total_models += 1
        
        if models_found:
            print(f"   Models found: {len(models_found)}")
            print(f"   Total size: {hub_size:.1f} MB ({hub_size/1024:.2f} GB)")
            
            # List each model
            for model_name, size, path in sorted(models_found, key=lambda x: x[1], reverse=True):
                print(f"     📦 {model_name}")
                print(f"        Size: {size:.1f} MB")
                print(f"        Path: {path}")
                
                # Check for specific files
                config_file = path / "refs" / "main"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            commit_hash = f.read().strip()
                            print(f"        Commit: {commit_hash[:8]}...")
                    except:
                        pass
                print()
            
            cache_locations.append(("Default HF Cache", hub_cache, hub_size))
            total_size += hub_size
        else:
            print("   No models found")
    
    # 2. Check transformers cache (alternative location)
    transformers_cache = default_cache / "transformers"
    print(f"\n2. Transformers Cache:")
    print(f"   Path: {transformers_cache}")
    print(f"   Exists: {'✅ YES' if transformers_cache.exists() else '❌ NO'}")
    
    if transformers_cache.exists() and any(transformers_cache.iterdir()):
        trans_size = get_folder_size_mb(transformers_cache)
        print(f"   Size: {trans_size:.1f} MB")
        cache_locations.append(("Transformers Cache", transformers_cache, trans_size))
        total_size += trans_size
    
    # 3. Check current project directory for model folders
    current_dir = Path.cwd()
    project_models = []
    
    print(f"\n3. Current Project Directory:")
    print(f"   Path: {current_dir}")
    
    # Look for common model folder names
    model_folders = ['models', 'model', 'cache', '.cache']
    
    for folder_name in model_folders:
        folder_path = current_dir / folder_name
        if folder_path.exists() and folder_path.is_dir():
            folder_size = get_folder_size_mb(folder_path)
            if folder_size > 10:  # Only show folders > 10MB
                project_models.append((folder_name, folder_path, folder_size))
    
    if project_models:
        print("   Model folders found:")
        for name, path, size in project_models:
            print(f"     📁 {name}/ - {size:.1f} MB")
            print(f"        Path: {path}")
            
            # List contents
            try:
                contents = list(path.iterdir())[:5]  # First 5 items
                for item in contents:
                    if item.is_dir():
                        print(f"          📁 {item.name}/")
                    else:
                        print(f"          📄 {item.name}")
                if len(list(path.iterdir())) > 5:
                    print(f"          ... and {len(list(path.iterdir())) - 5} more items")
            except:
                pass
            print()
            
            cache_locations.append((f"Project {name}", path, size))
            total_size += size
    else:
        print("   No model folders found")
    
    # 4. Check environment variables
    print(f"\n4. Environment Variables:")
    hf_home = os.environ.get('HF_HOME')
    hf_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
    transformers_cache_env = os.environ.get('TRANSFORMERS_CACHE')
    
    print(f"   HF_HOME: {hf_home or '❌ Not set'}")
    print(f"   HUGGINGFACE_HUB_CACHE: {hf_cache or '❌ Not set'}")
    print(f"   TRANSFORMERS_CACHE: {transformers_cache_env or '❌ Not set'}")
    
    # Check environment variable paths
    for env_var, env_path in [('HF_HOME', hf_home), ('HUGGINGFACE_HUB_CACHE', hf_cache), ('TRANSFORMERS_CACHE', transformers_cache_env)]:
        if env_path and os.path.exists(env_path):
            env_size = get_folder_size_mb(Path(env_path))
            if env_size > 10:
                print(f"   {env_var} exists with {env_size:.1f} MB")
                cache_locations.append((env_var, Path(env_path), env_size))
                total_size += env_size
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if cache_locations:
        print(f"Total cache locations found: {len(cache_locations)}")
        print(f"Total models: {total_models}")
        print(f"Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
        print("\nAll locations:")
        for name, path, size in cache_locations:
            print(f"  • {name}: {size:.1f} MB")
            print(f"    {path}")
    else:
        print("❌ No Hugging Face models found!")
        print("\nThis could mean:")
        print("  • Models haven't been downloaded yet")
        print("  • Models are in a different location")
        print("  • You're using a different caching method")
    
    print("\n" + "=" * 60)
    print("🛠️  ACTIONS YOU CAN TAKE")
    print("=" * 60)
    
    if total_size > 1000:  # More than 1GB
        print("💾 You have large model caches. Consider:")
        print("  • Removing unused models to free up space")
        print("  • Moving models to a different drive if space is limited")
    
    print("\n📋 To manage your cache:")
    print("  • Install HF CLI: pip install huggingface-hub[cli]")
    print("  • View cache: huggingface-cli scan-cache")
    print("  • Delete models: huggingface-cli delete-cache")
    
    print("\n🔍 To check specific model usage:")
    print("  • Look for import statements in your code")
    print("  • Check for model names like 'trocr', 'medieval', 'latin'")
    
    return cache_locations

def find_specific_models():
    """Find specific models you might be looking for"""
    print("\n" + "=" * 60)
    print("🎯 SEARCHING FOR SPECIFIC MODELS")
    print("=" * 60)
    
    # Models you might have downloaded
    target_models = [
        'medieval-data/trocr-medieval-latin-caroline',
        'magistermilitum/tridis_HTR',
        'microsoft/trocr-base-stage1',
        'microsoft/trocr-large-stage1',
        'medieval-data/trocr-medieval-humanistica'
    ]
    
    hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    
    found_models = []
    
    for model in target_models:
        # Convert model name to cache folder name
        cache_name = f"models--{model.replace('/', '--')}"
        model_path = hub_cache / cache_name
        
        if model_path.exists():
            size = get_folder_size_mb(model_path)
            found_models.append((model, model_path, size))
    
    if found_models:
        print("Found these specific models:")
        for model, path, size in found_models:
            print(f"  ✅ {model}")
            print(f"     Size: {size:.1f} MB")
            print(f"     Path: {path}")
            print()
        
        # Recommendations
        print("🎯 RECOMMENDATIONS:")
        for model, path, size in found_models:
            if 'trocr' in model.lower() and 'tridis' not in model.lower():
                print(f"  ❌ Consider removing {model} ({size:.1f} MB) - superseded by TRIDIS")
            elif 'tridis' in model.lower():
                print(f"  ✅ Keep {model} ({size:.1f} MB) - best for medieval Latin")
    else:
        print("❌ None of the target models found in cache")
    
    return found_models

if __name__ == "__main__":
    print("🔍 Hugging Face Model Finder")
    print("This script will find all downloaded Hugging Face models on your system")
    print()
    
    try:
        # Find all cache locations
        cache_locations = find_huggingface_cache()
        
        # Find specific models
        specific_models = find_specific_models()
        
        print("\n" + "=" * 60)
        print("✅ SEARCH COMPLETE")
        print("=" * 60)
        print(f"Script completed successfully!")
        print(f"Found {len(cache_locations)} cache locations")
        print(f"Check the output above for detailed information")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
