"""CLI Script: Generate and persist synthetic benchmark uplift dataset with known ground truth."""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import generate_synthetic_uplift_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic uplift dataset with known CATE ground truth.")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--treatment-ratio", type=float, default=0.85, help="Proportion of treated records")
    parser.add_argument("--observational", action="store_true", help="Introduce observational confounding bias")
    parser.add_argument("--output-path", type=str, default="data/synthetic/synthetic_uplift.csv", help="Target CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--base-cost", type=float, default=2.50, help="Base intervention cost")
    parser.add_argument("--base-margin", type=float, default=50.0, help="Base margin upon conversion")
    args = parser.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic dataset: N={args.n_samples}, treatment_ratio={args.treatment_ratio}, seed={args.seed}")
    df = generate_synthetic_uplift_data(
        n_samples=args.n_samples,
        treatment_ratio=args.treatment_ratio,
        observational_bias=args.observational,
        random_seed=args.seed,
        base_cost=args.base_cost,
        base_margin=args.base_margin,
    )

    df.to_csv(out_path, index=False)
    logger.info(f"Saved synthetic dataset successfully to {out_path} ({len(df)} records, {len(df.columns)} columns)")

    # Print summary of segment breakdown
    seg_counts = df["segment_label"].value_counts()
    logger.info("Customer Archetype Breakdown:")
    for seg, cnt in seg_counts.items():
        logger.info(f"  {seg}: {cnt} ({cnt / len(df) * 100:.2f}%)")

    logger.info(f"True Average Treatment Effect (ATE): {df['true_cate'].mean():.5f}")
    logger.info(f"Naive Observed ATE: {df.loc[df['treatment']==1, 'conversion'].mean() - df.loc[df['treatment']==0, 'conversion'].mean():.5f}")


if __name__ == "__main__":
    main()

