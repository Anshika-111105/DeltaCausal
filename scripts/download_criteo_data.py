"""Utility script to download or instruct setup for the full Criteo Uplift Dataset (v2.1).

Dataset Source: http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
File Size: ~311 MB compressed, ~3.5 GB uncompressed (~13.9M rows).
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
"""

import argparse
import gzip
import logging
from pathlib import Path
import shutil
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"


def download_criteo_dataset(output_dir: str = "data/raw", decompress: bool = False):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_target = out_dir / "criteo-research-uplift-v2.1.csv.gz"
    csv_target = out_dir / "criteo-research-uplift-v2.1.csv"

    if csv_target.exists():
        logger.info(f"Target CSV already exists at {csv_target}")
        return

    logger.info(f"Downloading Criteo Uplift v2.1 from {CRITEO_URL} to {gz_target}...")
    try:
        req = urllib.request.Request(CRITEO_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(gz_target, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        logger.info(f"Download complete: {gz_target}")

        if decompress:
            logger.info(f"Decompressing {gz_target} to {csv_target}...")
            with gzip.open(gz_target, "rb") as f_in, open(csv_target, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            logger.info("Decompression complete.")
    except Exception as e:
        logger.error(
            f"Automated download failed ({e}).\n"
            "If your environment lacks direct external network access, manually download the archive from:\n"
            f"  {CRITEO_URL}\n"
            f"and place it into: {csv_target.resolve()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Criteo Uplift dataset.")
    parser.add_argument("--output-dir", default="data/raw", help="Directory to save raw dataset")
    parser.add_argument("--decompress", action="store_true", help="Decompress .csv.gz to .csv")
    args = parser.parse_args()

    download_criteo_dataset(args.output_dir, args.decompress)

