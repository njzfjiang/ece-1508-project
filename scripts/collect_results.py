"""Collect and organize results from experiments."""

import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def collect_results(results_dir, output_file):
    """Collect results from all experiments.
    
    Args:
        results_dir: Directory containing experiment results
        output_file: File to save collected results
    """
    logger.info(f"Collecting results from {results_dir}")
    
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    # Collect results
    for model_dir in results_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            results[model_name] = {}
            logger.info(f"Processing model: {model_name}")
    
    # Save collected results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Collect experiment results')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--output', type=str, default='collected_results.json', help='Output file')
    
    args = parser.parse_args()
    collect_results(args.results_dir, args.output)


if __name__ == '__main__':
    main()
