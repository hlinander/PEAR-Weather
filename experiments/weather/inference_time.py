#!/usr/bin/env python
import sys
import importlib
import torch
import numpy as np
from pathlib import Path

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.serialization import deserialize_model, DeserializeConfig


from experiments.weather.data import DataHP

if __name__ == "__main__":
    device_id = ddp_setup()

    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    train_run = config_file.create_config(0, 10)

    ds_train = DataHP(train_run.train_config.train_data_config)
    dl_rmse = torch.utils.data.DataLoader(
        ds_train,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    epoch = int(sys.argv[2])

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

    model = deser_model.model
    model.eval()

    batch = next(iter(dl_rmse))

    surface_shape = batch["input_surface"].shape
    upper_shape = batch["input_upper"].shape

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        data_surface = torch.rand(surface_shape, device=device_id, dtype=torch.float32)
        data_upper = torch.rand(upper_shape, device=device_id, dtype=torch.float32)
        start.record()
        model(dict(input_surface=data_surface, input_upper=data_upper))
        end.record()
        torch.cuda.synchronize()

    times = []
    for _ in range(100):
        data_surface = torch.rand(surface_shape, device=device_id, dtype=torch.float32)
        data_upper = torch.rand(upper_shape, device=device_id, dtype=torch.float32)
        start.record()
        model(dict(input_surface=data_surface, input_upper=data_upper))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    print(times.mean(), times.std())
