import argparse
from pathlib import Path
from urllib.parse import urljoin
from tqdm.auto import tqdm
import pandas as pd
import requests


DATASET_ENDPOINTS = {
    "gtex": "https://brd.nci.nih.gov/brd/imagedownload/",
    "tcga": "https://api.gdc.cancer.gov/data/",
    # "panda" handled separately via Kaggle API
}


def download_http(base_url: str, remote_id: str, out_path: Path,
                  chunk_size: int = 8 * 1024 * 1024) -> bool:
    """
    Generic HTTP downloader with .part temp file support.
    (No resume logic here, but safe atomic rename.)
    """
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    url = urljoin(base_url, remote_id)

    try:
        with requests.get(url, stream=True, timeout=(5, 60)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        tmp.replace(out_path)
        return True
    except Exception as e:
        print(f"[HTTP] Failed {remote_id} -> {out_path.name}: {e}")
        # Clean up partial file if it exists
        if tmp.exists():
            tmp.unlink()
        return False


def download_panda(image_id: str, out_dir: Path) -> bool:
    """
    Download a single PANDA slide via Kaggle competition API.

    Assumes:
      - Competition: 'prostate-cancer-grade-assessment'
      - Remote file name: f'train_images/{image_id}'
      - Kaggle API credentials already configured (~/.kaggle/kaggle.json)
    """

    try:
        from kaggle import api as kaggle_api
    except ImportError:
        print("[PANDA] kaggle package not installed; cannot download PANDA data.")
        return False

    competition = "prostate-cancer-grade-assessment"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_tiff = out_dir / image_id
    if final_tiff.exists():
        tqdm.write(f"[PANDA] Skipping {image_id} (already exists)")
        return True

    remote_file = f"train_images/{image_id}"

    try:
        # Kaggle saves a it as zip
        kaggle_api.competition_download_file(
            competition=competition,
            file_name=remote_file,
            path=out_dir,
        )

        
        print(f"[PANDA] Done: {final_tiff}")
        return True

    except Exception as e:
        print(f"[PANDA] Failed {image_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download slides for MultiPathQA into per-dataset folders."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to MultiPathQA CSV.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Root output directory; one subfolder per benchmark will be created.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8 * 1024 * 1024,
        help="Download chunk size in bytes (default: 8 MiB).",
    )
    parser.add_argument(
        "--dataset",
        default="*",
        help=(
            "Which dataset/benchmark to download (e.g. tcga, gtex, panda, "
            "tcga_slidebench, tcga_expert_vqa). "
            "Use '*' to download all (default)."
        ),
    )

    args = parser.parse_args()
    csv_path = Path(args.csv)
    out_root = Path(args.out_root)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required_cols = ["benchmark_name", "image_path", "file_id"]
    df = df[required_cols].copy()
    df = df.drop_duplicates(subset=["benchmark_name", "image_path", "file_id"])

    total = len(df)
    success = 0
    fail = 0

    selected_dataset = args.dataset

    for benchmark, group in df.groupby("benchmark_name"):
        if selected_dataset != "*" and benchmark != selected_dataset:
            continue

        ds_root = out_root / benchmark
        ds_root.mkdir(parents=True, exist_ok=True)

        tqdm.write(f"=== Downloading dataset: {benchmark} (n={len(group)}) ===")


        for row in tqdm(
                group.itertuples(index=False),
                total=len(group),
                desc=benchmark,
                unit="slide",
            ):

            if benchmark == "panda":
                # PANDA via Kaggle; PANDA id is in file_id
                ok = download_panda(row.file_id, ds_root)

            else:
                # HTTP-based datasets
                if "tcga" in benchmark:
                    base_url = DATASET_ENDPOINTS["tcga"]
                if "gtex" in benchmark:
                    base_url = DATASET_ENDPOINTS["gtex"]

                # local output path: <out_root>/<benchmark>/<image_path>
                filename = row.image_path
                out_path = ds_root / filename

                part_path = out_path.with_suffix(out_path.suffix + ".part")
                if out_path.exists() or part_path.exists():
                    tqdm.write(f"[{benchmark}] Skipping {filename} (already exists or partial).")
                    success += 1  # counts as "handled"
                    continue
                ok = download_http(base_url, row.file_id, out_path, args.chunk_size)

            if ok:
                success += 1
            else:
                fail += 1

    print("\n=== Summary ===")
    print(f"Total rows in CSV:   {total}")
    print(f"Dataset filter:      {selected_dataset}")
    print(f"Successful downloads:{success}")
    print(f"Failed:              {fail}")
    print(f"Output root:         {out_root}")


if __name__ == "__main__":
    main()
