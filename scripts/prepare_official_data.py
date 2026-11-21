"""Fetch pinned official sources/data and verify archives before extraction."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import re
import tarfile
import time
import urllib.request
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
OFFICIAL = ROOT / "FloorSet"
GITHUB_SHA = "aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa"
DATASET_SHA = "7dc7c1bec07bcae583a274f5ed95e5be399cb8db"
FILES = {
    "validation": (
        "LiteTensorDataTest.tar.gz",
        69379249,
        "91535933f301f5bd3e3b00a7380f463d5eb654bfddc069ef5d1e5b9813854c6d",
    ),
    "training": (
        "LiteTensorData_v2.tar.gz",
        6627425876,
        "dad21c725a31185a826b0bac4c392e8c0d4de0f21d143b431ab70366f75daf82",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    url: str, target: Path, expected_size: int | None = None, expected_sha256: str | None = None
) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_size is not None and target.stat().st_size != expected_size:
            raise ValueError(f"Unexpected existing size: {target}")
        digest = sha256(target)
        if expected_sha256 is None or digest == expected_sha256:
            return {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": digest,
                "url": url,
            }
        raise ValueError(f"SHA-256 mismatch: {target}")
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "FloorSet-local-setup/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                size = expected_size or (offset + int(response.headers.get("Content-Length", 0)))
                mode = "ab" if offset else "wb"
                count = offset
                started = time.monotonic()
                last = started - 10
                with partial.open(mode) as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        count += len(chunk)
                        now = time.monotonic()
                        if now - last >= 10:
                            rate = (count - offset) / max(now - started, 0.001) / 1e6
                            print(
                                f"{target.name}: {count / 1e6:.1f}/{size / 1e6:.1f} MB ({rate:.1f} MB/s)",
                                flush=True,
                            )
                            last = now
                if expected_size is not None and count != expected_size:
                    raise IOError(f"Incomplete download: {count}/{expected_size}")
            digest = sha256(partial)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError(f"SHA-256 mismatch: {partial}")
            partial.replace(target)
            print(f"Verified {target.name}: sha256={digest}", flush=True)
            return {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": digest,
                "url": url,
            }
        except (OSError, TimeoutError) as exc:
            print(f"Retry {attempt}/5 ({type(exc).__name__}) for {target.name}", flush=True)
            if attempt == 5:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("Download did not complete")


def safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path outside destination: {name}")
    return target


def prepare_source() -> dict:
    record = fetch(
        f"https://api.github.com/repos/IntelLabs/FloorSet/zipball/{GITHUB_SHA}",
        DOWNLOADS / f"FloorSet-{GITHUB_SHA}.zip",
    )
    OFFICIAL.mkdir(exist_ok=True)
    with zipfile.ZipFile(record["path"]) as archive:
        prefix = archive.infolist()[0].filename.split("/")[0] + "/"
        for item in archive.infolist():
            relative = item.filename.removeprefix(prefix)
            if not relative:
                continue
            target = safe_target(OFFICIAL, relative)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    record["commit"] = GITHUB_SHA
    print(f"Official source ready: {OFFICIAL}", flush=True)
    return record


def prepare_dataset(kind: str) -> dict:
    name, size, digest = FILES[kind]
    url = f"https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/{DATASET_SHA}/{name}?download=true"
    record = fetch(url, DOWNLOADS / name, size, digest)
    count = 0
    total_size = 0
    top_dirs: set[str] = set()
    with tarfile.open(record["path"], "r|gz") as archive:
        for item in archive:
            target = safe_target(OFFICIAL, item.name)
            top_dirs.add(Path(item.name).parts[0])
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != item.size:
                    source = archive.extractfile(item)
                    if source is None:
                        raise IOError(item.name)
                    with source, target.open("wb") as dst:
                        shutil.copyfileobj(source, dst)
                count += 1
                total_size += item.size
                if count % 500 == 0:
                    print(f"{kind}: extracted {count} files", flush=True)
            else:
                raise ValueError(f"Unsupported archive member: {item.name}")
    record.update(
        dataset_revision=DATASET_SHA,
        extracted_files=count,
        extracted_bytes=total_size,
        top_directories=sorted(top_dirs),
    )
    print(f"{kind} ready: {count} files, {total_size / 1e9:.3f} GB", flush=True)
    return record


def prepare_torch() -> dict:
    index_url = "https://download.pytorch.org/whl/cu128/torch/"
    with urllib.request.urlopen(index_url, timeout=60) as response:
        content = response.read().decode("utf-8")
    candidates = [
        html.unescape(link)
        for link in re.findall(r'href="([^"]+)"', content)
        if "torch-2.7.1" in link and "cp312-cp312-win_amd64.whl" in link
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one pinned torch wheel, found {len(candidates)}")
    url, fragment = urllib.parse.urldefrag(urllib.parse.urljoin(index_url, candidates[0]))
    digest = urllib.parse.parse_qs(fragment)["sha256"][0]
    url = url.replace("download.pytorch.org/", "download-r2.pytorch.org/")
    name = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[1])
    return fetch(url, DOWNLOADS / name, expected_sha256=digest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["source", "validation", "training", "torch"])
    args = parser.parse_args()
    if args.kind == "source":
        record = prepare_source()
    elif args.kind == "torch":
        record = prepare_torch()
    else:
        record = prepare_dataset(args.kind)
    manifest = DOWNLOADS / f"{args.kind}_manifest.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
