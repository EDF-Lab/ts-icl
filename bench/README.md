## Benchmarks

This folder contains codes for reproducing TS-ICL's zero-shot performances on four extensive benchmarks:

1. **[Forecasting]** *FEV-BENCH* is a benchmark of 100 forecasting tasks across seven domains.
30 tasks include *known dynamic covariates*.
**Total: 235k windows** to forecast.
Details: [fev paper](https://arxiv.org/pdf/2509.26468).

2. **[Forecasting]** *TIME* is a benchmark of 98 forecasting tasks across 50 datasets and 9 domains.
**Total: 110k windows** to forecast.
Details: [TIME paper](https://arxiv.org/pdf/2602.12147).

3. **[Imputation]** *FM_IMPUTE* is a benchmark of resp. 33 univariate and 6 covariate-informed imputation datasets across 3 domains.
Four imputation settings are considered (2 pointwise + 2 block scenarios), resulting in a total of 156 tasks.
**Total: 1.3M windows** to impute.
Details: [fm_impute paper](https://openreview.net/pdf?id=cTk56KpsP5).

4. **[Imputation]** We adapt the *TIME* forecasting benchmark to cover univariate time series imputation.
Each lookback window of the forecasting benchmark is treated as a partially observed sequence with synthetically introduced missing values.
Four imputation settings are considered (2 pointwise + 2 block scenarios), resulting in a total of 392 tasks.
**Total: 440k windows** to impute.
Details: [TS-ICL paper](https://arxiv.org/abs/2606.05878).


### Installation

Running the scripts requires additionnal dependencies gathered in the `bench` group.
Install, e.g. with:

```bash
pip install tsicl[bench]
```

The installed dependencies are *datasets<4.0*, *fev>=0.7.0*, *gluonts>=0.16.2*.

### Guidelines

1. Download the [TS-ICL](https://huggingface.co/taharnbl/TS-ICL) checkpoint and store locally to `<TSICL_PATH>`.
2. Download benchmarks datasets from HuggingFace: [FEV_BENCH](https://huggingface.co/datasets/autogluon/fev_datasets), [TIME](https://huggingface.co/datasets/Real-TSF/TIME) and [FM_IMPUTE](https://huggingface.co/datasets/taharnbl/fm_impute_bench). Store them locally.
3. Create a `.env` file in the root directory of the project and add the following `keys`:
    - `TSICL_PATH`: path to TS-ICL's `.ckpt` file from step 1, e.g. `<TSICL_PATH>/tsicl-v1.ckpt`.
    - `FEV_BENCH_REPO`: path to *fev-bench* datasets downloaded at step 2.
    - `TIME_REPO`: path to *TIME* datasets downloaded at step 2.
    - `FM_IMPUTE_REPO`: path to *fm-impute-bench* datasets downloaded at step 2.

> `.env` follows the `key=value` syntax, similarly to a bash file.
4. Run the desired benchmarks using the scripts `run_<benchmark_name>.sh` in `bench/<benchmark_name>`

> Current scripts were written for running with `slurm`: adapt them (preamble, `srun` command, etc.) according to your own hardware requirements.