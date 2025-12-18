#!/usr/bin/env python
import os
import sys
import importlib
import torch
import numpy as np
from pathlib import Path

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.files import prepare_results
from lib.serialization import deserialize_model, DeserializeConfig

from lib.render_duck import (
    insert_artifact,
    insert_model_with_model_id,
    insert_checkpoint_pg,
    ensure_duck,
    attach_pg,
    sync,
)

from experiments.weather.data import Climatology, DataHP
from experiments.weather.metrics import (
    rmse_hp,
    rmse_dh,
    rmse_dh_on_dh,
    MeteorologicalData,
)


if __name__ == "__main__":
    device_id = ddp_setup()

    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    train_run = config_file.create_config(0, 10)
    lead_time_days = int(os.environ.get("LEADTIME", "1"))
    print("Lead time {lead_time}d")
    ds_train = DataHP(train_run.train_config.train_data_config)
    ds_rmse_config = (
        train_run.train_config.train_data_config.validation().with_lead_time_days(
            lead_time_days
        )
    )
    ds_rmse = DataHP(ds_rmse_config)
    dl_rmse = torch.utils.data.DataLoader(
        ds_rmse,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    ds_acc = Climatology(
        train_run.train_config.train_data_config.validation().with_lead_time_days(
            lead_time_days
        )
    )
    dl_acc = torch.utils.data.DataLoader(
        ds_acc,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    era5_meta = MeteorologicalData()

    epoch = int(sys.argv[2])

    print(f"[eval] Epoch {epoch}")
    deser_config = DeserializeConfig(
        train_run=create_ensemble_config(
            lambda eid: config_file.create_config(eid, epoch),
            1,
        ).members[0],
        device_id=device_id,
    )
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print("Can't deserialize")
        exit(0)

    insert_model_with_model_id(train_run, deser_model.model_id)

    result_path = prepare_results(
        f"{train_run.serialize_human()["run_id"]}",
        train_run,
    )

    def save_and_register(name, array):
        path = result_path / f"{name}.npy"

        np.save(
            path,
            array.detach().cpu().float().numpy(),
        )
        insert_artifact(deser_model.model_id, name, path, ".npy")

    ensure_duck(train_run)
    attach_pg()

    try:
        insert_checkpoint_pg(
            deser_model.model_id, int(epoch * len(ds_train)), "", db_prefix="pg."
        )
    except Exception as e:
        print(e)

    model = deser_model.model
    model.eval()

    print("[eval] rmse")
    if ds_rmse_config.driscoll_healy:
        rmse_res = rmse_dh(model, dl_rmse, device_id)
        rmse_res_on_dh_unweighted = rmse_dh_on_dh(
            model, dl_rmse, device_id, weighted=False
        )
        save_and_register(
            f"spatial_rmse_surface_e{epoch}_{lead_time_days}d_dh_unweighted.npydh",
            rmse_res_on_dh_unweighted.surface,
        )
        save_and_register(
            f"spatial_rmse_upper_e{epoch}_{lead_time_days}d_dh_unweighted.npydh",
            rmse_res_on_dh_unweighted.upper,
        )
    else:
        rmse_res = rmse_hp(model, dl_rmse, device_id)

    sync(train_run)
