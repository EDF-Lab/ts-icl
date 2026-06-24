#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fmimpute
#SBATCH --output=fmimpute.out
#SBATCH --error=fmimpute.err

CWD=$(pwd)
SCRIPT='bench/fmimpute'

if [[ "$CWD" =~ .*"$SCRIPT".* ]]; then
    cd ../../
fi

source .venv/bin/activate


# =================================================================
# User Config Section 
# =================================================================

# choose which bench to run:
run_univariate=true         # standard univariate benchmark
run_covar=true              # with covariates benchmark

# some settings:
context_len=4096            # max lookback length
num_tasks=null              # number of tasks to run, null -> all
batch_size=64               # batch size
make_plots=false            # whether to plot some forecasts
nb_plots=3                  # how many plots


# =================================================================
# End of User Config Section 
# =================================================================


# =================================================================
# Run inference scripts
# =================================================================

# univariate benchmark
if [ $run_univariate = true ]; then

    srun python3 -u bench/fmimpute/tsicl_univar.py    \
        "context_length=${context_len}"               \
        "num_tasks=${num_tasks}"                      \
        "batch_size=${batch_size}"                    \
        "++make_plots=${make_plots}"                  \
        "++nb_plots=${nb_plots}"

fi

# =================================================================

# covariate benchmark
if [ $run_covar = true ]; then

    # run with univariate target and no covariate:
    srun python3 -u bench/fmimpute/tsicl_covar.py     \
        "context_length=${context_len}"               \
        "num_tasks=${num_tasks}"                      \
        "batch_size=${batch_size}"                    \
        "use_covariates=false"                        \
        "++make_plots=false"                          \
        "++nb_plots=${nb_plots}"

    # run with sparse covariates:
    srun python3 -u bench/fmimpute/tsicl_covar.py     \
        "context_length=${context_len}"               \
        "num_tasks=${num_tasks}"                      \
        "batch_size=${batch_size}"                    \
        "use_covariates=true"                         \
        "++make_plots=false"                          \
        "++use_sparse_covariates=true"                \
        "++nb_plots=${nb_plots}"

    # run with covariates:
    srun python3 -u bench/fmimpute/tsicl_covar.py     \
        "context_length=${context_len}"               \
        "num_tasks=${num_tasks}"                      \
        "batch_size=${batch_size}"                    \
        "use_covariates=true"                         \
        "++make_plots=${make_plots}"                  \
        "++nb_plots=${nb_plots}"
    
fi

# =================================================================
