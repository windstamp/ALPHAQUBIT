#!/usr/bin/env python3
"""Check training status and duration on server."""

import os
import json
import subprocess
from datetime import datetime, timedelta

def check_training_status():
    """Check training status, duration, and progress."""
    
    print("=" * 70)
    print("TRAINING STATUS CHECK")
    print("=" * 70)
    
    # 1. Check running Python processes
    print("\n[1] Running Python Processes:")
    print("-" * 50)
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True
        )
        python_procs = [l for l in result.stdout.split('\n') if 'python' in l.lower() and 'grep' not in l]
        if python_procs:
            for proc in python_procs:
                print(proc)
        else:
            print("No Python processes running")
    except Exception as e:
        print(f"Error checking processes: {e}")
    
    # 2. Check log files
    print("\n[2] Log Files Status:")
    print("-" * 50)
    
    log_files = [
        "/root/work/ALPHAQUBIT/training.log",
        "/root/work/ALPHAQUBIT/training_full.log",
        "/root/work/ALPHAQUBIT/training_full_v2.log",
        "/root/work/ALPHAQUBIT/pipeline.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            stat = os.stat(log_file)
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)
            age = datetime.now() - mtime
            
            print(f"\n  {os.path.basename(log_file)}:")
            print(f"    Size: {size_kb:.1f} KB")
            print(f"    Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Age: {age}")
            
            # Get first and last lines with timestamps
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Find first timestamp
                        first_time = None
                        last_time = None
                        for line in lines[:100]:
                            if '2026-' in line or '2025-' in line:
                                try:
                                    ts = line.split()[0] + ' ' + line.split()[1].split(',')[0]
                                    first_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                                    break
                                except:
                                    pass
                        
                        for line in reversed(lines[-100:]):
                            if '2026-' in line or '2025-' in line:
                                try:
                                    ts = line.split()[0] + ' ' + line.split()[1].split(',')[0]
                                    last_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                                    break
                                except:
                                    pass
                        
                        if first_time and last_time:
                            duration = last_time - first_time
                            print(f"    Started: {first_time}")
                            print(f"    Last entry: {last_time}")
                            print(f"    Duration: {duration}")
                        
                        # Show last few lines
                        print(f"    Last 3 lines:")
                        for line in lines[-3:]:
                            print(f"      {line.strip()[:80]}")
            except Exception as e:
                print(f"    Error reading: {e}")
    
    # 3. Check results
    print("\n[3] Results Summary:")
    print("-" * 50)
    
    summary_file = "/root/work/ALPHAQUBIT/results/multi_npu_independent/summary.json"
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"  Summary found: {summary_file}")
            print(f"  Contents: {json.dumps(summary, indent=2)[:500]}")
        except Exception as e:
            print(f"  Error reading summary: {e}")
    else:
        print(f"  No summary file yet")
    
    # 4. Check NPU status
    print("\n[4] NPU Status:")
    print("-" * 50)
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True,
            timeout=10
        )
        print(result.stdout[:2000] if result.stdout else "No output")
    except FileNotFoundError:
        print("npu-smi not available (not on NPU server)")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_training_status()
