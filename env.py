from pathlib import Path
from lib.analytics_config import (
    AnalyticsConfig,
    StagingFilesystem,
    CentralDuckDB,
    CentralDuckLake,
)
from lib.compute_env_config import ComputeEnvironment, Paths

def get_analytics_config() -> AnalyticsConfig:
    return AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=Path("./staging"),
            archive_dir=Path("./archive"),
        ),
        central=CentralDuckDB(
            db_path=Path("./central.db")
        ),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )


def get_env() -> ComputeEnvironment:
    return ComputeEnvironment(
        paths=Paths(
            checkpoints=Path("./checkpoints"),
            datasets=Path("./datasets"),
            locks=Path("./locks"),
            distributed_requests=Path("./distributed_requests"),
            artifacts=Path("./artifacts"),
        ),
    )
