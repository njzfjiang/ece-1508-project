"""Prepare few-shot splits for experiments."""

import argparse
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def prepare_fewshot_splits(data_dir, output_dir, num_shots=10, num_seeds=5):
    """Prepare few-shot data splits.
    
    Args:
        data_dir: Input data directory
        output_dir: Output directory for splits
        num_shots: Number of shots for few-shot learning
        num_seeds: Number of different random seeds
    """
    logger.info(f"Preparing {num_shots}-shot splits with {num_seeds} seeds")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for seed in range(1, num_seeds + 1):
        split_dir = output_path / f"{num_shots}shot_seed{seed}"
        split_dir.mkdir(exist_ok=True)
        logger.info(f"Created split directory: {split_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare few-shot splits')
    parser.add_argument('--data_dir', type=str, required=True, help='Input data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_shots', type=int, default=10, help='Number of shots')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds')
    
    args = parser.parse_args()
    prepare_fewshot_splits(args.data_dir, args.output_dir, args.num_shots, args.num_seeds)


if __name__ == '__main__':
    main()
