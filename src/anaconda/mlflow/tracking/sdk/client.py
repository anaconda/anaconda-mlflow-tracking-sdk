""" This module provides an interface for MLFlow Tracking Operations. """

from typing import List, Optional, Union

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.pyfunc import PyFuncModel
from mlflow.store.entities import PagedList

from anaconda.enterprise.server.contracts import BaseModel


class AnacondaMlFlowClient(BaseModel):
    """
    This class provides an interface for interacting with the MLFlow Tracking Server.

    Attributes
    ----------
    client: MlflowClient
        An instance of a raw mlflow client.
    """

    client: MlflowClient

    def get_experiments(self, filter_string: Optional[str] = None) -> List[Experiment]:
        """
        Consumes the paged MLFlow experiments API and returns a consolidated list of found experiments.

        Returns
        -------
        experiments: list[Experiment]
            A list of `Experiment` objects found from the search.
        """

        experiments: PagedList[Experiment] = PagedList(items=[], token=None)

        halt_paging: bool = False
        page_token: Union[str, None] = None
        while not halt_paging:
            reported_experiments: PagedList[Experiment] = self.client.search_experiments(
                page_token=page_token, filter_string=filter_string
            )
            if reported_experiments.token is not None:
                page_token = reported_experiments.token
            else:
                halt_paging = True
            experiments += reported_experiments

        return list(experiments)

    def get_experiment_runs(self, experiment_id: str, filter_string: Optional[str] = None) -> List[Run]:
        """
        Consumes the paged MLFlow runs API and returns a consolidated list of found runs.

        Returns
        -------
        experiments: list[Run]
            A list of `Run` objects found from the search.
        """

        results: PagedList[Run] = PagedList(items=[], token=None)

        halt_paging: bool = False
        page_token: Union[str, None] = None
        while not halt_paging:
            reported_runs: PagedList[Run] = self.client.search_runs(
                experiment_ids=[experiment_id], page_token=page_token, filter_string=filter_string
            )
            if reported_runs.token is not None:
                page_token = reported_runs.token
            else:
                halt_paging = True
            results += reported_runs

        return list(results)

    def get_model_versions(self, model_name: str) -> PagedList[ModelVersion]:
        """
        Returns all model versions for the specified model name.

        Parameters
        ----------
        model_name: str
            The model name.

        Returns
        -------
        versions: PagedList[ModelVersion]
            A paged list of all model versions for the specified model.
        """

        return self.client.search_model_versions(filter_string=f"name='{model_name}'")

    def get_registered_models(self, filter_string: Optional[str] = None) -> List[RegisteredModel]:
        """
        Returns all registered models.

        Returns
        -------
        models: list[RegisteredModel]
            A list of registered models.
        """

        models: PagedList[RegisteredModel] = PagedList(items=[], token=None)

        halt_paging: bool = False
        page_token: Union[str, None] = None
        while not halt_paging:
            reported_models: PagedList[RegisteredModel] = self.client.search_registered_models(
                page_token=page_token, filter_string=filter_string
            )

            # This endpoint behaves differently then others, it appears that the token will be missing or simply the
            # empty string "".

            if reported_models.token is not None and reported_models.token != "":
                page_token = reported_models.token
            else:
                halt_paging = True
            models += reported_models

        return list(models)

    @staticmethod
    def load_model_by_version(name: str, version: int) -> PyFuncModel:
        """
        Creates a URL to load the model by version and returns the model.

        Parameters
        ----------
        name: Name of the model.
        version: The model version.

        Returns
        -------
        model: PyFuncModel
            Returns a python function based MLFlow model.
        """

        model_uri: str = f"models:/{name}/{version}"
        return AnacondaMlFlowClient.load_model_by_run(logged_model_rui=model_uri)

    @staticmethod
    def load_model_by_run(logged_model_rui: str) -> PyFuncModel:
        """
        Loads a tracked model from the provided URI.

        Parameters
        ----------
        logged_model_rui: Complete MLFlow Tracking Server Model URI

        Returns
        -------
        model: PyFuncModel
            Returns a python function based MLFlow model.
        """
        return mlflow.pyfunc.load_model(model_uri=logged_model_rui)

    @staticmethod
    def load_model_by_stage(name: str, stage: str) -> PyFuncModel:
        """
        Creates a URL to load the model by stage and returns the model.

        Parameters
        ----------
        name: Name of the model.
        stage: The model stage.

        Returns
        -------
        model: PyFuncModel
            Returns a python function based MLFlow model.
        """

        model_uri: str = f"models:/{name}/{stage}"
        return AnacondaMlFlowClient.load_model_by_run(logged_model_rui=model_uri)
