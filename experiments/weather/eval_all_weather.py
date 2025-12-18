import subprocess
import os

models = [
    "experiments/weather/persisted_configs/pear.py",
    "experiments/weather/persisted_configs/pangu.py",
    "experiments/weather/persisted_configs/pangu_large.py",
]

env = os.environ.copy()

for epoch in range(0, 250, 10):
    for lead_time in range(10):
        for model in models:
            cmd = [
                "./run.sh",
                "experiments/weather/evaluate.py",
                model,
                f"{epoch}",
            ]
            subprocess.run(cmd, env=env)
