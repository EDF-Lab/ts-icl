.. TS-ICL documentation master file, created by
   sphinx-quickstart on Thu Jun 11 11:51:40 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================================================================================
TS-ICL : A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning
======================================================================================

|arXiv| |PyPI| |Python|

TS-ICL is a continuous probabilistic Time Series Foundation Model (TSFM) that unifies forecasting and imputation in a single zero-shot architecture, requiring no task-specific training or fine-tuning.

.. toctree::
    :maxdepth: 1 
    :hidden:

    Quickstart <readme>
    Forecasting <pages/01_get_started_forecasting>
    Imputation <pages/02_get_started_imputation>
    API Reference </autoapi/index.rst>

.. raw:: html

    <style>
        .bd-main .bd-content .bd-article-container{
            max-width:80%;
        }
        .vertical-legend-timestep,  
        .vertical-legend-customer {
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            padding: 10px 5px;
            font-size: 1.2em;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #f8f9fa;
            border-right: 1px solid #dee2e6;
        }
        .vertical-legend-timestep {
            padding: 25px 5px;
        }
    </style>

.. raw:: html

   <br>

.. grid:: 1 1 1 4
    :gutter: 2 2 2 2

    .. grid-item-card::
        :img-top: _static/index_getting_started.svg
        :text-align: center

        Quickstart
        ^^^

        Check out the TS-ICL on-boarding guide.

        +++

        .. button-ref:: /src/0_quickstart.rst
            :expand:
            :color: secondary
            :click-parent:

            To the main quickstart

    .. grid-item-card::
        :img-top: _static/forecast-GFC17-covar_nolegend.png
        :text-align: center

        Forecasting
        ^^^

        Check out the TS-ICL forecasting guide.

        +++

        .. button-ref:: pages/01_get_started_forecasting
            :expand:
            :color: secondary
            :click-parent:

            To the forecasting quickstart

    .. grid-item-card::
        :img-top: _static/impute-covid19-energy_nolegend.png
        :text-align: center

        Imputation
        ^^^

        Check out the TS-ICL imputation guide.

        +++

        .. button-ref:: pages/02_get_started_imputation
            :expand:
            :color: secondary
            :click-parent:

            To the imputation quickstart

    .. grid-item-card::
        :img-top: _static/index_api.svg
        :text-align: center

        API reference
        ^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in TS-ICL.

        +++

        .. button-ref:: autoapi/index
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

.. raw:: html
   
   <br>

**Paper:** `TS-ICL: A Flexible Time-Indexed Foundation Model for Time
Series via In-Context Learning <https://arxiv.org/abs/2606.05878>`__