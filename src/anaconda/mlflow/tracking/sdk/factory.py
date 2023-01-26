""" Helper factory functions """
from mlflow import MlflowClient

from anaconda.enterprise.server.common.sdk import demand_env_var


def build_mlflow_client() -> MlflowClient:
    """
    Constructs a mlflow client with our globally controlled configuration options.

    Returns
    -------
    client: MlflowClient
        Instance of an MlflowClient.
    """

    return MlflowClient(
        tracking_uri=demand_env_var(name="MLFLOW_TRACKING_URI"), registry_uri=demand_env_var(name="MLFLOW_REGISTRY_URI")
    )
