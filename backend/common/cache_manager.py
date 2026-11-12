import time
import os

from modules.utils.logger import get_logger
from modules.utils.paths import BACKEND_CACHE_DIR


logger = get_logger()


def cleanup_old_files(cache_dir: str = BACKEND_CACHE_DIR, ttl: int = 60):
    now = time.time()
    place_holder_name = "cached_files_are_generated_here"
    for root, dirs, files in os.walk(cache_dir):
        for filename in files:
            if filename == place_holder_name:
                continue
            filepath = os.path.join(root, filename)
            try:
                modified = os.path.getmtime(filepath)
            except OSError:
                continue
            if now - modified > ttl:
                try:
                    os.remove(filepath)
                except FileNotFoundError:
                    continue
                except OSError as e:
                    logger.error(f"Error removing {filepath}: {e}")
