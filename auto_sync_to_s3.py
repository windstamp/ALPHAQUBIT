# =============================================================================
# Auto-Sync: Local → S3 → Server
# Run this script on your local Windows machine
# It watches for file changes and syncs to S3 automatically
# =============================================================================
#
# Usage: python auto_sync_to_s3.py
#
# Requirements: pip install watchdog
# =============================================================================

import time
import subprocess
import sys
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

# Configuration
LOCAL_PATH = r"C:\Users\Lenovo\software\ALPHAQUBIT"
S3_REMOTE = "nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT"
SYNC_DELAY = 5  # seconds to wait after last change before syncing
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    "*.egg-info",
    "backup_*",
    "*.npy",
    ".vs",
]

class SyncHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_change = 0
        self.pending_sync = False
    
    def should_ignore(self, path):
        for pattern in EXCLUDE_PATTERNS:
            if pattern.replace("*", "") in path:
                return True
        return False
    
    def on_any_event(self, event):
        if event.is_directory:
            return
        if self.should_ignore(event.src_path):
            return
        
        self.last_change = time.time()
        self.pending_sync = True
        print(f"📝 Changed: {event.src_path}")

def build_exclude_args():
    args = []
    for pattern in EXCLUDE_PATTERNS:
        args.extend(["--exclude", pattern])
    return args

def sync_to_s3():
    print("\n🔄 Syncing to S3...")
    cmd = ["rclone", "copy", LOCAL_PATH, S3_REMOTE, "--progress"] + build_exclude_args()
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ Sync complete!\n")
    else:
        print("❌ Sync failed!\n")

def main():
    print("=" * 60)
    print("  Auto-Sync: Local → S3")
    print("=" * 60)
    print(f"📁 Watching: {LOCAL_PATH}")
    print(f"☁️  S3 Target: {S3_REMOTE}")
    print(f"⏱️  Sync delay: {SYNC_DELAY}s after last change")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    handler = SyncHandler()
    observer = Observer()
    observer.schedule(handler, LOCAL_PATH, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
            # Sync if there are pending changes and enough time has passed
            if handler.pending_sync and (time.time() - handler.last_change) > SYNC_DELAY:
                sync_to_s3()
                handler.pending_sync = False
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
