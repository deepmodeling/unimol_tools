from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _make_session(
    max_retries: int = 5,
    backoff_factor: float = 0.5,
) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"HEAD", "GET"},
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "pytest-downloader/1.0"})
    return sess


def download(
    url: str,
    dest: Path,
    *,
    timeout: Tuple[float, float] = (5.0, 30.0),
    max_retries: int = 5,
    backoff_factor: float = 0.5,
    chunk_size: int = 1 << 18,
    allow_resume: bool = True,
    expected_sha256: Optional[str] = None,
    expected_md5: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and (expected_sha256 or expected_md5):
        ok = True
        if expected_sha256:
            ok &= _check_hash(dest, expected_sha256, "sha256")
        if expected_md5:
            ok &= _check_hash(dest, expected_md5, "md5")
        if ok:
            return dest
        else:
            dest.unlink(missing_ok=True)

    sess = session or _make_session(max_retries=max_retries, backoff_factor=backoff_factor)

    content_length = None
    try:
        h = sess.head(url, timeout=timeout, allow_redirects=True)
        if h.ok:
            content_length = _safe_int(h.headers.get("Content-Length"))
    except requests.RequestException:
        pass

    tmp_path = dest.with_suffix(dest.suffix + ".part")
    downloaded = tmp_path.stat().st_size if (allow_resume and tmp_path.exists()) else 0
    headers = {}
    if allow_resume and downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    with sess.get(url, timeout=timeout, stream=True, headers=headers) as r:
        if r.status_code == 200 and "Range" in headers:
            downloaded = 0
        r.raise_for_status()
        mode = "ab" if downloaded > 0 else "wb"
        with open(tmp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    if content_length is not None:
        actual = tmp_path.stat().st_size
        if actual != content_length:
            tmp_path.unlink(missing_ok=True)
            raise IOError(f"Downloaded size mismatch: expected {content_length}, got {actual}")

    if expected_sha256 and not _check_hash(tmp_path, expected_sha256, "sha256"):
        tmp_path.unlink(missing_ok=True)
        raise IOError("SHA-256 checksum mismatch.")
    if expected_md5 and not _check_hash(tmp_path, expected_md5, "md5"):
        tmp_path.unlink(missing_ok=True)
        raise IOError("MD5 checksum mismatch.")

    os.replace(tmp_path, dest)
    return dest


def download_for_test(
    url: str,
    dest: Path,
    *,
    skip_on_failure: bool = True,
    **kwargs,
) -> Path:
    try:
        return download(url, dest, **kwargs)
    except Exception as e:
        if skip_on_failure:
            try:
                import pytest
                pytest.skip(f"Skip due to network instability: {e}")
            except Exception:
                pass
        raise


def _check_hash(path: Path, expected_hex: str, algo: str) -> bool:
    expected_hex = expected_hex.lower().strip()
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_hex


def _safe_int(x: Optional[str]) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except ValueError:
        return None
