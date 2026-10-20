"""
Live log syncer for FloraNet-250 pipeline
Mirrors live task output into reports/mega_pipeline.log so the user can watch progress in their editor.
"""
import time
import shutil
from pathlib import Path

SRC_LOG = Path(r"C:\Users\User\.gemini\antigravity\brain\d97fd749-7a53-48bf-99ed-b47c211057f2\.system_generated\tasks\task-1969.log")
DST_LOG = Path(r"c:\Users\User\Desktop\FYP\CoreTech\Plant_Identification_Experiment\reports\mega_pipeline.log")

DST_LOG.parent.mkdir(parents=True, exist_ok=True)

print(f"[LogSyncer] Started mirroring {SRC_LOG.name} -> {DST_LOG}...")
last_size = 0

for _ in range(300): # max 30-40 minutes
    if SRC_LOG.exists():
        try:
            curr_size = SRC_LOG.stat().st_size
            if curr_size != last_size:
                shutil.copyfile(SRC_LOG, DST_LOG)
                last_size = curr_size
        except Exception:
            pass
    time.sleep(8)

print("[LogSyncer] Finished syncing.")

