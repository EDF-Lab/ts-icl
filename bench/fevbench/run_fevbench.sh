#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=fev
#SBATCH --output=fev.out
#SBATCH --error=fev.err

CWD=$(pwd)
SCRIPT='bench/fevbench/'

if [[ "$CWD" =~ .*"$SCRIPT".* ]]; then
    cd ../../
fi

source .venv/bin/activate


# =================================================================
# User Config Section 
# =================================================================


# choose which FEV bench to run:
run_bench=true              # main FEV benchmark
run_missing=true            # with missing values in the lookback

# some settings:
context_len=4096            # max lookback length
num_tasks=null              # number of tasks to run, null -> all
make_plots=false            # whether to plot some forecasts
nb_plots=3                  # how many plots


# =================================================================
# End of User Config Section 
# =================================================================


# =================================================================
# Run inference scripts
# =================================================================

# towards leaderboard
if [ $run_bench = true ]; then

    srun python3 -u bench/fevbench/tsicl_main.py      \
        "context_length=${context_len}"               \
        "num_tasks=${num_tasks}"                      \
        "to_univariate=false"                         \
        "use_covariates=true"                         \
        "use_static_covariates=false"                 \
        "past_only=false"                             \
        "++make_plots=${make_plots}"                  \
        "++nb_plots=${nb_plots}"

fi

# =================================================================

# univar withing missing values benchmark
if [ $run_missing = true ]; then
    
    base_seed=42
    srun python3 -u bench/fevbench/tsicl_missing_values.py  \
        "seed=${base_seed}"                                 \
        "context_length=${context_len}"                     \
        "num_tasks=${num_tasks}"                            \
        "to_univariate=true"                                \
        "use_covariates=false"                              \
        "use_static_covariates=false"

fi

# =================================================================