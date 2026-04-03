from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    dataset_id: str | None = None
    version: int | None = None
    archive_name: str | None = None
    git_url: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "jmuben": DatasetSpec(
        name="jmuben",
        source="mendeley",
        dataset_id="t2r6rszp5c",
        version=1,
        archive_name="t2r6rszp5c-1.zip",
    ),
    "jmuben2": DatasetSpec(
        name="jmuben2",
        source="mendeley",
        dataset_id="tgv3zb82nd",
        version=1,
        archive_name="tgv3zb82nd-1.zip",
    ),
    "bracol": DatasetSpec(
        name="bracol",
        source="mendeley",
        dataset_id="yy2k5y8mxg",
        version=1,
        archive_name="yy2k5y8mxg-1.zip",
    ),
    "rocole": DatasetSpec(
        name="rocole",
        source="mendeley",
        dataset_id="c5yvn32dzg",
        version=2,
        archive_name="c5yvn32dzg-2.zip",
    ),
    "uganda": DatasetSpec(
        name="uganda",
        source="mendeley",
        dataset_id="k36wnd6knb",
        version=1,
        archive_name="k36wnd6knb-1.zip",
    ),
    "rust_miner_brazil": DatasetSpec(
        name="rust_miner_brazil",
        source="mendeley",
        dataset_id="vfxf4trtcg",
        version=5,
        archive_name="vfxf4trtcg-5.zip",
    ),
    "clr_eafit": DatasetSpec(
        name="clr_eafit",
        source="git",
        git_url="https://github.com/dvelaren/clr-dataset",
    ),
}

SESSION = requests.Session()


def fetch_mendeley_zip_metadata(spec: DatasetSpec) -> dict[str, object]:
    if not spec.dataset_id or spec.version is None:
        raise ValueError(f"Dataset spec for {spec.name} is incomplete.")
    api_url = (
        f"https://data.mendeley.com/api/datasets-v2/datasets/"
        f"{spec.dataset_id}/zip?version={spec.version}"
    )
    response = SESSION.get(api_url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "url" not in payload:
        raise RuntimeError(f"Mendeley did not return a zip URL for {spec.name}.")
    return payload


def fetch_mendeley_folders(spec: DatasetSpec) -> list[dict[str, object]]:
    if not spec.dataset_id or spec.version is None:
        raise ValueError(f"Dataset spec for {spec.name} is incomplete.")
    api_url = (
        f"https://data.mendeley.com/public-api/datasets/"
        f"{spec.dataset_id}/folders/{spec.version}"
    )
    response = SESSION.get(
        api_url,
        timeout=60,
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
    )
    response.raise_for_status()
    return response.json()


def fetch_mendeley_files(
    spec: DatasetSpec,
    folder_id: str = "root",
) -> list[dict[str, object]]:
    if not spec.dataset_id or spec.version is None:
        raise ValueError(f"Dataset spec for {spec.name} is incomplete.")
    api_url = (
        f"https://data.mendeley.com/public-api/datasets/"
        f"{spec.dataset_id}/files?folder_id={folder_id}&version={spec.version}"
    )
    response = SESSION.get(
        api_url,
        timeout=60,
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
    )
    response.raise_for_status()
    return response.json()


def download_file(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        return

    print(f"[download] {destination}")
    with SESSION.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_public_file(file_item: dict[str, object], destination: Path) -> Path:
    content = file_item["content_details"]
    download_url = content["download_url"]
    expected_size = int(content["size"]) if content.get("size") else None
    download_file(download_url, destination, expected_size=expected_size)
    return destination


def extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted_from"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == archive_path.name:
        print(f"[skip] already extracted: {archive_path.name}")
        return

    if not zipfile.is_zipfile(archive_path):
        raise zipfile.BadZipFile(f"{archive_path} is not a valid zip archive.")

    print(f"[extract] {archive_path} -> {destination}")
    with zipfile.ZipFile(archive_path) as zipped:
        zipped.extractall(destination)
    marker.write_text(archive_path.name, encoding="utf-8")


def build_folder_paths(folders: list[dict[str, object]]) -> dict[str, Path]:
    by_id = {folder["id"]: folder for folder in folders if folder.get("id")}
    resolved: dict[str, Path] = {}

    def resolve(folder_id: str) -> Path:
        if folder_id in resolved:
            return resolved[folder_id]
        folder = by_id[folder_id]
        parts = [str(folder["name"]).strip()]
        parent_id = folder.get("parent_id")
        while parent_id and parent_id in by_id:
            parent = by_id[parent_id]
            parts.append(str(parent["name"]).strip())
            parent_id = parent.get("parent_id")
        parts.reverse()
        resolved[folder_id] = Path(*parts)
        return resolved[folder_id]

    for folder_id in by_id:
        resolve(folder_id)
    return resolved


def iter_mendeley_targets(spec: DatasetSpec) -> list[tuple[Path, dict[str, object]]]:
    folders = fetch_mendeley_folders(spec)
    folder_paths = build_folder_paths(folders)
    targets: list[tuple[Path, dict[str, object]]] = []

    for file_item in fetch_mendeley_files(spec, "root"):
        targets.append((Path(file_item["filename"]), file_item))

    for folder_id, relative_path in folder_paths.items():
        files = fetch_mendeley_files(spec, folder_id)
        for file_item in files:
            targets.append((relative_path / str(file_item["filename"]), file_item))

    return targets


def should_extract(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def download_mendeley_dataset(
    spec: DatasetSpec,
    dataset_root: Path,
    archives_dir: Path,
    extract_archives: bool,
) -> dict[str, object]:
    targets = iter_mendeley_targets(spec)
    manifest_files: list[dict[str, object]] = []
    archive_jobs: list[tuple[Path, dict[str, object]]] = []
    direct_jobs: list[tuple[Path, dict[str, object]]] = []

    for relative_path, file_item in targets:
        filename = str(file_item["filename"])
        if should_extract(Path(filename)):
            archive_jobs.append((relative_path, file_item))
        else:
            direct_jobs.append((relative_path, file_item))

    for relative_path, file_item in archive_jobs:
        archive_destination = archives_dir / spec.name / relative_path
        downloaded_path = download_public_file(file_item, archive_destination)
        if extract_archives:
            extract_target = dataset_root / relative_path.with_suffix("")
            extract_zip(downloaded_path, extract_target)
        manifest_files.append(
            {
                "relative_path": str(relative_path),
                "folder_id": file_item.get("folder_id"),
                "download_url": file_item["content_details"]["download_url"],
                "size": file_item.get("size"),
            }
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {}
        for relative_path, file_item in direct_jobs:
            target_destination = dataset_root / relative_path
            future = executor.submit(download_public_file, file_item, target_destination)
            future_map[future] = (relative_path, file_item)

        for future in as_completed(future_map):
            relative_path, file_item = future_map[future]
            future.result()
            manifest_files.append(
                {
                    "relative_path": str(relative_path),
                    "folder_id": file_item.get("folder_id"),
                    "download_url": file_item["content_details"]["download_url"],
                    "size": file_item.get("size"),
                }
            )

    return {
        "source": "mendeley",
        "dataset_id": spec.dataset_id,
        "version": spec.version,
        "path": str(dataset_root),
        "file_count": len(manifest_files),
        "files": manifest_files,
    }


def clone_or_update_repo(spec: DatasetSpec, destination: Path) -> None:
    if not spec.git_url:
        raise ValueError(f"Dataset spec for {spec.name} has no git_url.")
    git_dir = destination / ".git"
    if git_dir.exists():
        import subprocess

        subprocess.run(
            ["git", "-C", str(destination), "pull", "--ff-only"],
            check=True,
        )
        return

    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"Destination {destination} exists and is not empty; refusing to clone over it."
        )

    if destination.exists():
        shutil.rmtree(destination)

    import subprocess

    subprocess.run(
        ["git", "clone", "--depth", "1", spec.git_url, str(destination)],
        check=True,
    )


def resolve_targets(requested: list[str]) -> list[DatasetSpec]:
    if not requested or requested == ["all"]:
        return list(DATASETS.values())
    unknown = sorted(set(requested) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")
    return [DATASETS[name] for name in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and optionally extract the public datasets used by RoyaDestroyer-AI."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        help=f"Dataset names or 'all'. Available: {', '.join(sorted(DATASETS))}",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives but do not extract them.",
    )
    parser.add_argument(
        "--archives-dir",
        type=Path,
        help="Optional directory where raw archives are stored. Defaults to data/raw/_archives.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    raw_root = settings.data_root / "raw"
    archives_dir = args.archives_dir or (raw_root / "_archives")
    targets = resolve_targets(args.datasets)

    manifest: dict[str, dict[str, object]] = {}
    errors: dict[str, str] = {}

    for spec in targets:
        dataset_root = raw_root / spec.name
        try:
            if spec.source == "git":
                clone_or_update_repo(spec, dataset_root)
                manifest[spec.name] = {
                    "source": "git",
                    "git_url": spec.git_url,
                    "path": str(dataset_root),
                }
                print(f"[ok] {spec.name}")
                continue

            manifest[spec.name] = download_mendeley_dataset(
                spec,
                dataset_root=dataset_root,
                archives_dir=archives_dir,
                extract_archives=not args.no_extract,
            )
            print(f"[ok] {spec.name}")
        except Exception as exc:
            errors[spec.name] = str(exc)
            manifest[spec.name] = {
                "source": spec.source,
                "path": str(dataset_root),
                "error": str(exc),
            }
            print(f"[error] {spec.name}: {exc}")

    manifest_path = settings.data_root / "reports" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")
    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
