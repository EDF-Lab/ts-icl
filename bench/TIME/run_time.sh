#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=time
#SBATCH --output=time.out
#SBATCH --error=time.err

CWD=$(pwd)
SCRIPT='bench/TIME'

if [[ "$CWD" =~ .*"$SCRIPT".* ]]; then
    cd ../../
fi

source .venv/bin/activate

# =================================================================
# User Config Section 
# =================================================================

# choose which benchmark to run:
run_forecasting=true         # univariate forecast
run_imputation=true          # univariate imputation

# choose all datasets:
datasets="all_datasets"

# some settings:
context_len=4096            # max lookback length
batch_size=64
make_plots=false            # whether to plot some forecasts
nb_plots=3                  # how many plots

# =================================================================
# End of User Config Section 
# =================================================================


# =================================================================
# Run inference scripts
# =================================================================


# time forecasting full multivar benchmark
if [ $run_forecasting = true ]; then
    srun python3 -u bench/TIME/tsicl_forecasting.py  \
        "context_length=${context_len}"              \
        "datasets=${datasets}"
    
fi

# =================================================================

# time imputation full multivar benchmark
if [ $run_imputation = true ]; then
    srun python3 -u bench/TIME/tsicl_imputation.py   \
        "context_length=${context_len}"              \
        "datasets=${datasets}"                       \
        "make_plots=${make_plots}"                   \
        "nb_plots=${nb_plots}"
fi

# =================================================================
