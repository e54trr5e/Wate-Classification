"""Legacy entry point. All model families now use shared_dataset."""
import subprocess
import sys
from pathlib import Path

if __name__=="__main__":
    script=Path(__file__).resolve().with_name("00_prepare_shared_dataset.py")
    print("This legacy YOLO-only split has been replaced by the common dataset split.")
    print("Running 00_prepare_shared_dataset.py so all models receive identical images.")
    raise SystemExit(subprocess.call([sys.executable,str(script),*sys.argv[1:]]))
