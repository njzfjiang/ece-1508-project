#!/bin/bash
# Script to download datasets

set -e

echo "Downloading dataset..."

# Create data directory if it doesn't exist
mkdir -p data/raw

# Add your download commands here
# Example:
# wget <url> -O data/raw/<filename>

echo "Download complete!"
