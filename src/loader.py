"""
loader.py — Reads input files, streams JSONL line-by-line.
"""

import gzip
import json
import os
import re
from typing import Generator, Dict, Any, Tuple

from utils import log, ts, get_file_size_mb, progress_bar


def load_job_description(input_dir: str) -> str:
    """
    Parse job_description.md, strip Markdown, return plain text.
    """
    path = os.path.join(input_dir, "job_description.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[LOADER] job_description.md not found at {path}")

    log("LOADER", f"Loading job description from {path} ({get_file_size_mb(path):.2f} MB)")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    plain = _strip_markdown(raw)
    word_count = len(plain.split())
    log("LOADER", f"Job description loaded — {word_count} words after markdown stripping")
    
    # --- FIX: Validate that we actually got text ---
    if len(plain.strip()) < 200:
        raise ValueError(
            f"Job description text is too short ({len(plain)} characters). "
            "Check that job_description.md is not empty and that _strip_markdown is not over-stripping."
        )
    return plain


def _strip_markdown(text: str) -> str:
    """Remove Markdown formatting, return clean plain text."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"(\*{1,3}|_{1,3})(.*?)\1", r"\2", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>+\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"[-]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_candidates_path(input_dir: str) -> Tuple[str, bool]:
    """
    Find candidates file. Prefers .jsonl.gz, falls back to .jsonl.
    Returns (path, is_gzipped).
    """
    gz_path = os.path.join(input_dir, "candidates.jsonl.gz")
    jsonl_path = os.path.join(input_dir, "candidates.jsonl")

    if os.path.exists(gz_path):
        log("LOADER", f"Found gzipped candidates file: {gz_path} ({get_file_size_mb(gz_path):.2f} MB)")
        return gz_path, True
    elif os.path.exists(jsonl_path):
        log("LOADER", f"Found plain JSONL file: {jsonl_path} ({get_file_size_mb(jsonl_path):.2f} MB)")
        return jsonl_path, False
    else:
        raise FileNotFoundError(
            f"[LOADER] No candidates file found in {input_dir}. "
            "Expected candidates.jsonl.gz or candidates.jsonl"
        )


def estimate_line_count(path: str, is_gzipped: bool, sample_bytes: int = 1024 * 512) -> int:
    """
    Estimate line count from a sample of the file.
    For gzipped files, reads the first chunk of uncompressed data.
    """
    try:
        opener = gzip.open if is_gzipped else open
        mode = "rt" if is_gzipped else "r"
        with opener(path, mode, encoding="utf-8") as f:
            chunk = f.read(sample_bytes)
        lines_in_sample = chunk.count("\n")
        if is_gzipped:
            total_compressed = os.path.getsize(path)
            ratio = 4.5
            total_bytes_est = total_compressed * ratio
        else:
            total_bytes_est = os.path.getsize(path)
        bytes_per_line = sample_bytes / max(lines_in_sample, 1)
        return int(total_bytes_est / bytes_per_line)
    except Exception:
        return 100_000


def stream_candidates(
    input_dir: str,
    log_every: int = 5000,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream candidates JSONL file line-by-line. Yields one parsed dict per candidate.
    Logs progress periodically.
    """
    path, is_gzipped = resolve_candidates_path(input_dir)
    file_size_mb = get_file_size_mb(path)
    estimated_total = estimate_line_count(path, is_gzipped)

    log("LOADER", f"File size: {file_size_mb:.2f} MB | Estimated candidates: ~{estimated_total:,}")
    log("LOADER", "Starting streaming of candidates line-by-line...")

    opener = gzip.open if is_gzipped else open
    mode = "rt" if is_gzipped else "r"
    count = 0
    errors = 0

    with opener(path, mode, encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
                count += 1
                if count % log_every == 0:
                    bar = progress_bar(count, estimated_total)
                    print(
                        f"[{ts()}] [LOADER] Streaming candidates: {bar}",
                        flush=True,
                    )
                yield record
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    log("LOADER", f"Skipping malformed line {count + errors}: {e}", level="WARN")

    log("LOADER", f"Streaming complete — {count:,} candidates loaded, {errors} parse errors")