#!/usr/bin/env python
"""
Pull Server Results from S3 to Local Machine

Usage:
    python pull_from_s3.py              # Pull latest results
    python pull_from_s3.py --list       # List available results on S3
    python pull_from_s3.py --all        # Pull all results
    python pull_from_s3.py --include-models  # Include model weights
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Configuration
S3_REMOTE = "nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
PROJECT_NAME = "ALPHAQUBIT"
S3_RESULTS_PATH = f"{S3_REMOTE}/{PROJECT_NAME}/server_results"
LOCAL_RESULTS_DIR = Path("./s3_results")

# Exclude patterns for model files
EXCLUDE_PATTERNS = [
    "*.pth", "*.pt", "*.bin", "*.safetensors", 
    "*.ckpt", "*.h5", "*.npy", "*.npz"
]


def check_rclone():
    """Check if rclone is installed."""
    try:
        result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def list_s3_contents():
    """List contents on S3."""
    print("=" * 60)
    print("  S3 Contents")
    print("=" * 60)
    print(f"S3 Path: {S3_RESULTS_PATH}")
    print()
    
    # List files
    cmd = ["rclone", "ls", S3_RESULTS_PATH]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error listing S3: {result.stderr}")
        return
    
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []
    
    if not files or files == ['']:
        print("📭 No files found on S3")
        return
    
    # Categorize files
    json_files = []
    log_files = []
    plot_files = []
    model_files = []
    other_files = []
    
    for line in files:
        if not line.strip():
            continue
        parts = line.strip().split(None, 1)
        if len(parts) >= 2:
            size, name = parts
            size_mb = int(size) / (1024 * 1024)
            
            if name.endswith('.json'):
                json_files.append((name, size_mb))
            elif name.endswith(('.log', '.txt', '.md')):
                log_files.append((name, size_mb))
            elif name.endswith(('.png', '.pdf', '.svg')):
                plot_files.append((name, size_mb))
            elif name.endswith(('.pth', '.pt', '.bin', '.ckpt', '.h5')):
                model_files.append((name, size_mb))
            else:
                other_files.append((name, size_mb))
    
    print("📊 Results Files:")
    for name, size in json_files[:20]:
        print(f"  📄 {name} ({size:.2f} MB)")
    
    print("\n📝 Log Files:")
    for name, size in log_files[:10]:
        print(f"  📃 {name} ({size:.2f} MB)")
    
    print("\n📈 Plot Files:")
    for name, size in plot_files[:10]:
        print(f"  🖼️  {name} ({size:.2f} MB)")
    
    if model_files:
        print("\n🧠 Model Files (excluded by default):")
        for name, size in model_files[:5]:
            print(f"  💾 {name} ({size:.2f} MB)")
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"  JSON files:  {len(json_files)}")
    print(f"  Log files:   {len(log_files)}")
    print(f"  Plot files:  {len(plot_files)}")
    print(f"  Model files: {len(model_files)}")
    print(f"  Other files: {len(other_files)}")
    print("=" * 60)


def pull_from_s3(include_models=False, specific_pattern=None):
    """Pull results from S3 to local."""
    print("=" * 60)
    print("  Pull Results from S3")
    print("=" * 60)
    print(f"S3 Source: {S3_RESULTS_PATH}")
    print(f"Local Destination: {LOCAL_RESULTS_DIR}")
    print()
    
    # Create local directory
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = ["rclone", "copy", S3_RESULTS_PATH, str(LOCAL_RESULTS_DIR), "--progress"]
    
    # Add exclude patterns if not including models
    if not include_models:
        print("⚠️  Excluding model files (use --include-models to include)")
        for pattern in EXCLUDE_PATTERNS:
            cmd.extend(["--exclude", pattern])
    else:
        print("📦 Including model files (this may take longer)")
    
    # Add specific pattern filter if provided
    if specific_pattern:
        cmd.extend(["--include", specific_pattern])
        print(f"🔍 Filtering: {specific_pattern}")
    
    print()
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Execute
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("✅ Pull complete!")
        print()
        
        # Show what was downloaded
        show_local_contents()
    else:
        print()
        print("❌ Pull failed!")
        return False
    
    return True


def show_local_contents():
    """Show downloaded contents."""
    print("📁 Downloaded files:")
    print("-" * 40)
    
    if not LOCAL_RESULTS_DIR.exists():
        print("  (no files)")
        return
    
    for f in sorted(LOCAL_RESULTS_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            rel_path = f.relative_to(LOCAL_RESULTS_DIR)
            print(f"  {rel_path} ({size_kb:.1f} KB)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Pull server results from S3")
    parser.add_argument("--list", action="store_true", help="List S3 contents")
    parser.add_argument("--include-models", action="store_true", help="Include model files")
    parser.add_argument("--pattern", type=str, help="Filter pattern (e.g., '*.json')")
    parser.add_argument("--show-local", action="store_true", help="Show local downloaded files")
    
    args = parser.parse_args()
    
    # Check rclone
    if not check_rclone():
        print("❌ rclone not found!")
        print("Install from: https://rclone.org/downloads/")
        print("Or with: winget install Rclone.Rclone")
        sys.exit(1)
    
    if args.list:
        list_s3_contents()
    elif args.show_local:
        show_local_contents()
    else:
        pull_from_s3(
            include_models=args.include_models,
            specific_pattern=args.pattern
        )


if __name__ == "__main__":
    main()
