# scripts/time_utils.py
from datetime import datetime
import pytz

def seconds_until_target_hour(target_hour=8, target_min=0, tz=pytz.UTC):
    now = datetime.now(tz)
    target = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
    if now >= target: return 0
    return max(0, int((target - now).total_seconds()))
