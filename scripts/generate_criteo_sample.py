"""Generate a realistic Criteo Uplift v2.1 format raw sample dataset in data/raw/criteo_sample.csv.

Matches the official 16-column schema:
f0 through f11, treatment, conversion, visit, exposure.
Treatment ratio: ~85% treated, ~15% control.
Conversion rate: ~4.5%.
"""

from pathlib import Path
import sys

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.synthetic_generator import generate_synthetic_uplift_data


def generate_criteo_sample(
    n_samples: int = 50000,
    output_path: str = "data/raw/criteo_sample.csv",
    random_seed: int = 42,
):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate using realistic underlying process
    df_raw = generate_synthetic_uplift_data(
        n_samples=n_samples,
        treatment_ratio=0.85,
        observational_bias=False,
        random_seed=random_seed,
    )

    # In Criteo's raw format, only the 16 standard columns are present
    criteo_columns = [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]
    df_criteo = df_raw[criteo_columns].copy()

    df_criteo.to_csv(out_file, index=False)
    print(f"Generated raw Criteo-format dataset at {out_file} ({len(df_criteo):,} records, {len(criteo_columns)} columns).")
    print("Schema preview:")
    print(df_criteo.head(2))


if __name__ == "__main__":
    generate_criteo_sample()

