"""
Download Google Quantum AI Experiment Data from Zenodo

This script downloads the experimental data for "Quantum error correction 
below the surface code threshold" from Zenodo.

Source: https://zenodo.org/records/13273331
DOI: 10.5281/zenodo.13273331

The data includes:
- google_105Q_surface_code_d3_d5_d7.zip (5.7 GB) - 105 Qubit Surface Code
- google_72Q_surface_code_d3_d5_set1.zip (30.2 GB) - 72 Qubit Surface Code Set 1
- google_72Q_surface_code_d3_d5_set2.zip (11.6 GB) - 72 Qubit Surface Code Set 2
- google_72Q_repetition_code_d29.zip (65.0 GB) - 72 Qubit Repetition Code

Total: 112.5 GB

Usage:
    python download_google_experiment_data.py
    python download_google_experiment_data.py --file 105Q  # Download only 105Q data
    python download_google_experiment_data.py --extract    # Extract after download
"""

import os
import sys
import argparse
import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import time

# Zenodo base URL
ZENODO_BASE = "https://zenodo.org/records/13273331/files"

# Files to download with metadata
FILES = {
    "105Q": {
        "name": "google_105Q_surface_code_d3_d5_d7.zip",
        "size_gb": 5.7,
        "size_bytes": 6_119_587_456,  # Approximate
        "md5": "21fa6ad35b395d838ebcdbc92e364a12",
        "description": "105 Qubit Surface Code (d3, d5, d7)"
    },
    "72Q_set1": {
        "name": "google_72Q_surface_code_d3_d5_set1.zip",
        "size_gb": 30.2,
        "size_bytes": 32_428_982_272,  # Approximate
        "md5": "7875d0fa37fab89e37cf5cd305114cef",
        "description": "72 Qubit Surface Code (d3, d5) Set 1 - Primary dataset"
    },
    "72Q_set2": {
        "name": "google_72Q_surface_code_d3_d5_set2.zip",
        "size_gb": 11.6,
        "size_bytes": 12_455_567_360,  # Approximate
        "md5": "cb331869b9fe7e95d6ec9c945d5278a6",
        "description": "72 Qubit Surface Code (d3, d5) Set 2"
    },
    "repetition": {
        "name": "google_72Q_repetition_code_d29.zip",
        "size_gb": 65.0,
        "size_bytes": 69_793_218_560,  # Approximate
        "md5": "12c2cc1e5a924c273342b2cb5654394d",
        "description": "72 Qubit Repetition Code (d29) - Large dataset"
    }
}

# Output directory
OUTPUT_DIR = Path(__file__).parent / "google_experiment_data"


def format_size(size_bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def download_progress(count, block_size, total_size):
    """Progress callback for urlretrieve."""
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    downloaded = count * block_size
    bar_len = 50
    filled = int(bar_len * percent / 100)
    bar = '=' * filled + '-' * (bar_len - filled)
    sys.stdout.write(f'\r  [{bar}] {percent}% ({format_size(downloaded)}/{format_size(total_size)})')
    sys.stdout.flush()


def verify_md5(filepath, expected_md5):
    """Verify MD5 checksum of a file."""
    print(f"  Verifying MD5 checksum...")
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    actual_md5 = md5_hash.hexdigest()
    if actual_md5 == expected_md5:
        print(f"  ✓ MD5 verified: {actual_md5}")
        return True
    else:
        print(f"  ✗ MD5 mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False


def download_file(file_key, output_dir, verify=True):
    """Download a single file from Zenodo."""
    if file_key not in FILES:
        print(f"Unknown file key: {file_key}")
        return False
    
    file_info = FILES[file_key]
    filename = file_info["name"]
    filepath = output_dir / filename
    url = f"{ZENODO_BASE}/{filename}?download=1"
    
    print(f"\n{'='*70}")
    print(f"File: {filename}")
    print(f"Size: {file_info['size_gb']} GB")
    print(f"Description: {file_info['description']}")
    print(f"URL: {url}")
    
    # Check if file already exists
    if filepath.exists():
        existing_size = filepath.stat().st_size
        expected_size = file_info["size_bytes"]
        if existing_size >= expected_size * 0.99:  # Allow 1% tolerance
            print(f"Status: File already exists ({format_size(existing_size)})")
            if verify:
                verify_md5(filepath, file_info["md5"])
            return True
        else:
            print(f"Status: Partial download exists ({format_size(existing_size)}), re-downloading...")
    
    print(f"Downloading...")
    start_time = time.time()
    
    try:
        urlretrieve(url, filepath, reporthook=download_progress)
        print()  # New line after progress bar
        
        elapsed = time.time() - start_time
        file_size = filepath.stat().st_size
        speed = file_size / elapsed / (1024 * 1024)
        print(f"  Downloaded {format_size(file_size)} in {elapsed:.1f}s ({speed:.2f} MB/s)")
        
        if verify:
            verify_md5(filepath, file_info["md5"])
        
        print(f"  ✓ Download complete!")
        return True
        
    except URLError as e:
        print(f"\n  ✗ Download failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n  Download interrupted by user")
        return False


def extract_zip(filepath, extract_to):
    """Extract a zip file."""
    print(f"Extracting {filepath.name}...")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  ✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Google Quantum AI experiment data from Zenodo"
    )
    parser.add_argument(
        "--file", 
        choices=list(FILES.keys()) + ["all", "surface_code"],
        default="surface_code",
        help="Which file(s) to download. 'surface_code' downloads all surface code data (recommended). Default: surface_code"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for downloads. Default: {OUTPUT_DIR}"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract zip files after download"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 verification"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files and exit"
    )
    
    args = parser.parse_args()
    
    # List files and exit
    if args.list:
        print("\nAvailable files:")
        print("-" * 70)
        total_size = 0
        for key, info in FILES.items():
            print(f"  {key:12} - {info['name']}")
            print(f"               {info['size_gb']} GB - {info['description']}")
            total_size += info['size_gb']
        print("-" * 70)
        print(f"  Total: {total_size:.1f} GB")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Google Quantum AI Experiment Data Downloader")
    print("Source: https://zenodo.org/records/13273331")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Determine which files to download
    if args.file == "all":
        files_to_download = list(FILES.keys())
    elif args.file == "surface_code":
        files_to_download = ["105Q", "72Q_set1", "72Q_set2"]
    else:
        files_to_download = [args.file]
    
    total_size = sum(FILES[f]["size_gb"] for f in files_to_download)
    print(f"\nFiles to download: {', '.join(files_to_download)}")
    print(f"Total size: {total_size:.1f} GB")
    print(f"Verify checksums: {not args.no_verify}")
    
    # Download files
    downloaded = []
    failed = []
    
    for file_key in files_to_download:
        success = download_file(file_key, args.output_dir, verify=not args.no_verify)
        if success:
            downloaded.append(file_key)
        else:
            failed.append(file_key)
    
    # Extract if requested
    if args.extract and downloaded:
        print(f"\n{'='*70}")
        print("Extracting files...")
        for file_key in downloaded:
            filepath = args.output_dir / FILES[file_key]["name"]
            extract_zip(filepath, args.output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Downloaded: {len(downloaded)}/{len(files_to_download)} files")
    if downloaded:
        print(f"  ✓ {', '.join(downloaded)}")
    if failed:
        print(f"  ✗ Failed: {', '.join(failed)}")
    
    print(f"\nFiles location: {args.output_dir}")
    
    if downloaded and not args.extract:
        print("\nTo extract the files, run:")
        print(f"  python {Path(__file__).name} --extract")
        print("  or manually:")
        for file_key in downloaded:
            print(f"    Expand-Archive {args.output_dir / FILES[file_key]['name']} -DestinationPath {args.output_dir}")
    
    print(f"\nTo use with AlphaQubit:")
    print(f"  python run_create_all_samples.py {args.output_dir} --shots 1000")
    print(f"  python make_all_pretraining_noise.py --experiment-root {args.output_dir}")


if __name__ == "__main__":
    main()
