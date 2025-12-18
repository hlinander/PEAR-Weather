# 🍐 PEAR: Equal Area Weather Forecasting on the Sphere
Official code for

["PEAR: Equal Area Weather Forecasting on the Sphere"](https://arxiv.org/abs/2505.17720)

by *Hampus Linander, Christoffer Petersson, Daniel Persson, Jan E. Gerken*.

## Abstract
Machine learning methods for global medium-range weather forecasting have
recently received immense attention. Following the publication of the Pangu
Weather model, the first deep learning model to outperform traditional numerical
simulations of the atmosphere, numerous models have been published in this
domain, building on Pangu's success. However, all of these models operate on
input data and produce predictions on the Driscoll--Healy discretization of
the sphere which suffers from a much finer grid at the poles than around the
equator. In contrast, in the Hierarchical Equal Area iso-Latitude Pixelization
(HEALPix) of the sphere, each pixel covers the same surface area, removing
unphysical biases. Motivated by a growing support for this grid in meteorology
and climate sciences, we propose to perform weather forecasting with deep
learning models which natively operate on the HEALPix grid. To this end, we
introduce Pangu Equal ARea (PEAR), a transformer-based weather forecasting model
which operates directly on HEALPix-features and outperforms the corresponding
model on Driscoll--Healy without any computational overhead.

## UV
Dependencies are specified through `pyproject.toml`
```
uv venv
uv sync
````

## Configure paths
Edit `env.py` to match where you want to store the dataset, checkpoints etc.

## Prepare dataset
PEAR and baselines are trained on a subset of [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete).

Make sure that CDSAPI is configured
[https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)

Download the training years 2007-2017 and the validation year 2019:
```
  uv run ./experiments/weather/download_lite.py
  uv run ./experiments/weather/download_lite_eval.py
  uv run ./experiments/weather/hydrate_dataset_cache.py
```

## Training model

Train PEAR
```
  ./run.sh experiments/weather/persisted_configs/train_nside64_single_relpos_pad_and_patch.py
```

Train Pangu models
```
  ./run.sh experiments/weather/persisted_configs/train_pangu_nside64_adapted.py
  ./run.sh experiments/weather/persisted_configs/train_pangu_nside64.py
```

## Evaluate checkpoints

To evaluate the epoch 100 checkpoint for the trained PEAR model (and similarly for the Pangu models)
```
  ./run.sh ./experiments/weather/evaluate.py ./experiments/weather/persisted_configs/train_nside64_single_relpos_pad_and_patch.py 100
```

To evaluate every 10th epoch for all models
```
  ./run.sh ./experiments/weather/eval_all_weather.py
```

## Consolidate metrics
Training and evaluation outputs parquet files to the staging path specified in `env.py`.
To merge these into a single duckdb file for further processing use
```
  ./run.sh ./ingestion/ingest.py
```
