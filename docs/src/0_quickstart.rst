.. container::

Quickstart
============

**Paper:** `TS-ICL: A Flexible Time-Indexed Foundation Model for Time
Series via In-Context Learning <https://arxiv.org/abs/2606.05878>`__

**TS-ICL** is a continuous probabilistic Time Series Foundation Model
(TSFM) that unifies **forecasting** and **imputation** in a single
zero-shot architecture, requiring no task-specific training or
fine-tuning.

.. image:: /contents/TS-ICL-v1.png
   :width: 70%
   :align: center
   :alt: alt


Installation
------------

.. code-block:: bash

   pip install tsicl

**Requirements:** Python ≥ 3.12, PyTorch ≥ 2.5.1


Usage
-----------

.. code-block:: python

   from tsicl import TSICL

   model = TSICL(model_path="checkpoints/tsicl-v1.ckpt")

   # Forecasting — predict the next 96 timesteps
   point, quantiles = model.forecast(
       inputs            = my_series,        # e.g. 1-D numpy array or tensor
       prediction_length = 96,
       quantile_levels   = [0.1, 0.5, 0.9],
       denormalize       = True
   )

   # Imputation — reconstruct NaN values
   point, quantiles = model.impute(
       inputs          = my_series_with_nans,
       quantile_levels = [0.1, 0.5, 0.9],
       denormalize     = True
   )

Both methods return a ``(point_prediction, quantile_predictions)``
tuple. **NaN values are handled natively** — no preprocessing required.

--------------

Notebooks
---------

Step-by-step tutorials on synthetic Gaussian Processes:

+---------------------------------------+-----------------------------------+
| Notebook                              | Description                       |
+=======================================+===================================+
| `get_started_imputation.ipynb         | Pointwise & block missingness,    |
| <pages/02_get_started_imputation>`__  | covariate-aware imputation,       |
|                                       | output format, batch processing   |
+---------------------------------------+-----------------------------------+
| `get_started_forecasting.ipynb        | Univariate forecasting, partially |
| <pages/01_get_started_forecasting>`__ | observed look-back,               |
|                                       | covariate-aware forecasting,      |
|                                       | batch processing                  |
+---------------------------------------+-----------------------------------+


--------------

Model
-----

TS-ICL processes each time series through four successive modules:

1. **Time Series Encoder** — a Perceiver-like architecture that
   compresses observed (timestamp, value) pairs into M = 32 learnable
   latent tokens via cross-attention. Accepts inputs of **arbitrary
   length** without preprocessing.
2. **Channel Mixer** — aggregates information across channels via
   cross-attention. Selectively integrates covariate representations
   into the target’s representation when covariates are provided.
3. **Temporal Context Query Module** — maps any query timestamp to a
   context-aware embedding using Fourier (NeRF-style) positional
   encoding. Enables prediction at arbitrary timestamps, including on
   **irregular grids**.
4. **In-Context Regressor** — a causal Transformer that reads observed
   (representation, value) pairs as in-context training examples and
   outputs **99 quantiles** at the queried timestamps.

A single checkpoint (``tsicl-v1.ckpt``) contains two specialised
components — one trained with masking for imputation, one with causal
masking for forecasting — sharing the same architecture backbone.

--------------

Performance
-----------

Forecasting — ``fev-bench``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

TS-ICL is highly competitive with the best forecasting foundation
models, while being **fast** at inference. TS-ICL **efficently leverages
covariate** (when relevant) and is also **robust to sparse look-back
windows**.

.. image:: /contents/fevbench.png
   :width: 70%
   :align: center
   :alt: alt

*Example — pointwise forecast with a known covariate (GFC17 dataset):*

.. image:: /contents/forecast-GFC17-covar.png
   :width: 70%
   :align: center
   :alt: alt

Forecasting — ``TIME benchmark``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /contents/Time-Benchmark.png
   :width: 70%
   :align: center
   :alt: alt

Imputation — ``fm-impute-bench``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TS-ICL achieves **state-of-the-art imputation** across 132 univariate
and 24 covariate-aware tasks, outperforming the best tabular foundation
model baseline while being **~50× faster** at inference.

.. image:: /contents/fm-impute-bench.png
   :width: 70%
   :align: center
   :alt: alt

*Example — block imputation with uncertainty quantification (COVID-19
energy dataset):*

.. image:: /contents/impute-covid19-energy.png
   :width: 70%
   :align: center
   :alt: alt

--------------

Citation
--------

If you use TS-ICL for research purposes, please consider citing the
associated paper:

.. code-block:: bibtex

   @article{lenaour2026tsicl,
     title={TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning},
     author={Le Naour, Etienne and Nabil, Tahar and Petralia, Adrien},
     journal={arXiv preprint arXiv:2606.05878},
     year={2026}
   }

--------------

Contributors
------------

-  `Etienne Le Naour <https://github.com/EtienneLnr>`__
-  `Tahar Nabil <https://github.com/TaharNbl>`__
-  `Adrien Petralia <https://github.com/adrienpetralia>`__
-  Marc Héry (EDF)

--------------

License
-------

TS-ICL weights and code are released under a non-commercial license, see
`LICENSE <LICENSE>`__.

Contact
-------

To learn more or request a commercial license, please contact us at:
tsicl-contact_at_edf.fr (replace *at* with @).

.. |arXiv| image:: https://img.shields.io/badge/arXiv-2606.05878-b31b1b.svg
   :target: https://arxiv.org/abs/2606.05878
.. |PyPI| image:: https://img.shields.io/pypi/v/tsicl.svg
   :target: https://pypi.org/project/tsicl
.. |Python| image:: https://img.shields.io/badge/python-%3E%3D3.12-blue.svg
   :target: https://www.python.org/
