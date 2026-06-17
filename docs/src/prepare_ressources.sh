#!/usr/bin/env bash
set -e

# remove notebooks if needed
rm -f docs/src/01_get_started_forecasting.ipynb
rm -f docs/src/02_get_started_imputation.ipynb

# Copy notebooks
cp notebooks/get_started_forecasting.ipynb docs/src/01_get_started_forecasting.ipynb
cp notebooks/get_started_imputation.ipynb docs/src/02_get_started_imputation.ipynb

# Adapt paths to doc
for nb in docs/src/*.ipynb; do
  sed -i 's|\.\./docs/contents/|../contents/|g' "$nb"
done

# Copy readme in docs
cp README.md docs/readme.md

echo "Copy and update figure paths of doc notebooks done."