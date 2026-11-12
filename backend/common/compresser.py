import os
import tempfile
import zipfile
from typing import List, Optional
import hashlib

from modules.utils.logger import get_logger


logger = get_logger()


def compress_files(file_paths: List[str], output_zip_path: str) -> str:
    """
    Compress multiple files into a single zip file.

    Args:
    file_paths (List[str]): List of paths to files to be compressed.
    output_zip (str): Path and name of the output zip file.

    Raises:
    FileNotFoundError: If any of the input files doesn't exist.
    """
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    compression = zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(output_zip_path, 'w', compression=compression) as zipf:
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_name = os.path.basename(file_path)
            zipf.write(file_path, file_name)
    return output_zip_path


def create_download_archive(file_paths: List[str], cache_dir: str, prefix: str) -> str:
    """Zip `file_paths` into a unique archive inside `cache_dir`.

    Every download request gets its own file, so concurrent requests can never
    share (and truncate) one archive. The temporary file is removed if zipping
    fails; the caller removes it after the response is sent.
    """
    fd, archive_path = tempfile.mkstemp(dir=cache_dir, prefix=f"{prefix}-", suffix=".zip")
    os.close(fd)
    try:
        compress_files(file_paths, archive_path)
    except Exception:
        try:
            os.remove(archive_path)
        except OSError:
            pass
        raise
    return archive_path


def get_file_hash(file_path: str) -> str:
    """Generate the hash of a file using the specified hashing algorithm. It generates hash by content not path.

    Raises OSError when the file is missing or unreadable: an error string must
    never be stored as a SHA-256 artifact identifier.
    """
    hash_func = hashlib.new("sha256")
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def find_file_by_hash(dir_path: str, hash_str: str) -> Optional[str]:
    """Get file path from the directory based on its hash.

    Raises FileNotFoundError when the directory is absent; files that vanish or
    cannot be read while scanning are skipped, not reported as matches.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist")

    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if not os.path.isfile(path):
            continue
        try:
            if get_file_hash(path) == hash_str:
                return path
        except OSError as e:
            logger.warning(f"Skipping {path} during hash lookup: {e}")
    return None


def resolve_cache_artifact(cache_dir: str, relative_path: str) -> Optional[str]:
    """Resolve a cache-relative artifact path, refusing anything outside the cache."""
    if not relative_path:
        return None
    root = os.path.realpath(cache_dir)
    resolved = os.path.realpath(os.path.join(root, relative_path))
    if resolved != root and not resolved.startswith(root + os.sep):
        return None
    return resolved
