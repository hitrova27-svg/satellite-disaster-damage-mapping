"""Resumable multi-connection downloader for the Kaggle xBD archive.

The official Kaggle CLI uses one HTTP stream.  This helper splits the archive
into independent byte ranges so an interrupted run can resume each part.
It never stores or prints the signed download URL or Kaggle credentials.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
from tqdm import tqdm


OWNER = "qianlanzz"
DATASET = "xbd-dataset"
ARCHIVE = "xbd-dataset.zip"
CHUNK_SIZE = 1024 * 1024


def get_download_response():
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiDownloadDatasetRequest()
        request.owner_slug = OWNER
        request.dataset_slug = DATASET
        return client.datasets.dataset_api_client.download_dataset(request)


def remote_info() -> tuple[str, dict[str, str], int]:
    response = get_download_response()
    try:
        total = int(response.headers["Content-Length"])
        headers = dict(getattr(response.request, "headers", {}))
        return str(response.url), headers, total
    finally:
        response.close()


def download_part(
    index: int,
    part_path: Path,
    range_start: int,
    range_end: int,
    expected_total: int,
    progress: tqdm,
    retries: int,
) -> None:
    expected_size = range_end - range_start + 1
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing > expected_size:
        raise RuntimeError(f"{part_path} is larger than its expected byte range")
    if existing == expected_size:
        return

    for attempt in range(retries + 1):
        existing = part_path.stat().st_size if part_path.exists() else 0
        request_start = range_start + existing
        try:
            url, headers, remote_total = remote_info()
            if remote_total != expected_total:
                raise RuntimeError(
                    f"Remote size changed: expected {expected_total}, got {remote_total}"
                )
            headers["Range"] = f"bytes={request_start}-{range_end}"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Part {index}: expected HTTP 206, got {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {request_start}-"):
                    raise RuntimeError(
                        f"Part {index}: unexpected Content-Range {content_range!r}"
                    )
                part_path.parent.mkdir(parents=True, exist_ok=True)
                with part_path.open("ab") as output:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            output.write(chunk)
                            progress.update(len(chunk))

            actual = part_path.stat().st_size
            if actual != expected_size:
                raise RuntimeError(
                    f"Part {index}: expected {expected_size} bytes, got {actual}"
                )
            return
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 60))


def combine_parts(parts: list[Path], destination: Path, total: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as output, tqdm(
        total=total, desc="Combining", unit="B", unit_scale=True
    ) as progress:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(8 * CHUNK_SIZE):
                    output.write(chunk)
                    progress.update(len(chunk))
    if temporary.stat().st_size != total:
        raise RuntimeError("Combined archive has the wrong size")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--parts",
        type=int,
        default=8,
        help="Number of persistent byte-range files (independent of active workers).",
    )
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(ARCHIVE))
    parser.add_argument("--parts-dir", type=Path, default=Path(".download-parts"))
    parser.add_argument(
        "--adopt-partial",
        type=Path,
        help="Move an existing CLI partial archive into the first range part.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.parts < 1:
        raise SystemExit("--workers and --parts must both be at least 1")

    _, _, total = remote_info()
    part_size = math.ceil(total / args.parts)
    parts = [args.parts_dir / f"part-{i:02d}" for i in range(args.parts)]

    if args.adopt_partial and args.adopt_partial.exists():
        if parts[0].exists():
            raise SystemExit(f"Cannot adopt partial: {parts[0]} already exists")
        args.parts_dir.mkdir(parents=True, exist_ok=True)
        if args.adopt_partial.stat().st_size > part_size:
            raise SystemExit("Existing partial is larger than the first range part")
        shutil.move(str(args.adopt_partial), str(parts[0]))

    existing = sum(path.stat().st_size for path in parts if path.exists())
    print(
        f"Downloading {total / 1024**3:.2f} GiB in {args.parts} persistent ranges "
        f"using {args.workers} connections; "
        f"resuming at {existing / 1024**2:.1f} MiB"
    )
    with tqdm(
        total=total,
        initial=existing,
        desc="Downloading",
        unit="B",
        unit_scale=True,
    ) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for index, path in enumerate(parts):
                start = index * part_size
                end = min(total, start + part_size) - 1
                futures.append(
                    pool.submit(
                        download_part,
                        index,
                        path,
                        start,
                        end,
                        total,
                        progress,
                        args.retries,
                    )
                )
            for future in as_completed(futures):
                future.result()

    combine_parts(parts, args.output, total)
    print(f"Complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
