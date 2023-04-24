import unittest
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.store.entities import PagedList

from src.anaconda.mlflow.tracking.sdk import AnacondaMlFlowClient


class TestAnacondaMlFlowClient(unittest.TestCase):
    mock_counter: int = 1

    def setUp(self):
        self.maxDiff = None
        self.client: AnacondaMlFlowClient = AnacondaMlFlowClient(client=MlflowClient())

    def generate_mock_experiment(self) -> Experiment:
        experiment = Experiment(
            artifact_location="",
            creation_time=datetime.now(),
            experiment_id=str(self.mock_counter),
            last_update_time=datetime.now(),
            lifecycle_stage="",
            name=str(self.mock_counter),
            tags={},
        )
        self.mock_counter += 1
        return experiment

    def generate_mock_run(self) -> Run:
        run = Run(run_info={}, run_data={})
        self.mock_counter += 1
        return run

    def generate_mock_model_version(self) -> ModelVersion:
        model_version = ModelVersion(
            name=f"mock-model-{str(self.mock_counter)}",
            version=str(self.mock_counter),
            creation_timestamp=datetime.now(),
        )
        self.mock_counter += 1
        return model_version

    def generate_mock_registered_model(self) -> RegisteredModel:
        registered_model = RegisteredModel(name=str(self.mock_counter))
        self.mock_counter += 1
        return registered_model

    def test_anaconda_mlflow_client(self):
        mlflow_client: MlflowClient = MlflowClient()
        client: AnacondaMlFlowClient = AnacondaMlFlowClient(client=mlflow_client)
        self.assertIsInstance(obj=client, cls=AnacondaMlFlowClient)

    def test_get_experiments_simple(self):
        mock_client: MagicMock = MagicMock()
        mock_response_data = PagedList[Experiment](items=[self.generate_mock_experiment()], token=None)
        mock_client.search_experiments.return_value = mock_response_data

        self.client.client = mock_client
        response: List[Experiment] = self.client.get_experiments()
        self.assertEqual(response, mock_response_data)

    def test_get_experiments_is_paged(self):
        mock_response_data_one = PagedList[Experiment](items=[self.generate_mock_experiment()], token="token")
        mock_response_data_two = PagedList[Experiment](items=[self.generate_mock_experiment()], token=None)
        self.client.client = MagicMock()
        self.client.client.search_experiments.side_effect = [mock_response_data_one, mock_response_data_two]

        response: List[Experiment] = self.client.get_experiments()
        self.assertEqual(response, [mock_response_data_one[0], mock_response_data_two[0]])

    def test_get_experiment_runs_simple(self):
        mock_client: MagicMock = MagicMock()
        mock_response_data = PagedList[Run](items=[self.generate_mock_run()], token=None)
        mock_client.search_runs.return_value = mock_response_data

        self.client.client = mock_client
        response: List[Run] = self.client.get_experiment_runs(experiment_id="1")
        self.assertEqual(list(mock_response_data), response)

    def test_get_experiment_runs_simple_paged(self):
        mock_response_data_one = PagedList[Run](items=[self.generate_mock_run()], token="token")
        mock_response_data_two = PagedList[Run](items=[self.generate_mock_run()], token=None)
        self.client.client = MagicMock()
        self.client.client.search_runs.side_effect = [mock_response_data_one, mock_response_data_two]

        response: List[Run] = self.client.get_experiment_runs(experiment_id="1")
        self.assertEqual(response, [mock_response_data_one[0], mock_response_data_two[0]])

    def test_get_model_versions(self):
        mock_client: MagicMock = MagicMock()
        mock_return_value = PagedList[ModelVersion](items=[self.generate_mock_model_version()], token=None)
        mock_client.search_model_versions.return_value = mock_return_value

        self.client.client = mock_client
        results: PagedList[ModelVersion] = self.client.get_model_versions(model_name="mock-model")

        self.assertEqual(list(mock_return_value), list(results))
        mock_client.search_model_versions.assert_called_with(filter_string="name='mock-model'")

    def test_get_registered_models_simple(self):
        mock_response_data = PagedList[RegisteredModel](items=[self.generate_mock_registered_model()], token=None)
        self.client.client = MagicMock()
        self.client.client.search_registered_models.return_value = mock_response_data

        response: List[RegisteredModel] = self.client.get_registered_models()
        self.assertEqual(response, list(mock_response_data))

    def test_get_registered_models_paged(self):
        mock_response_data_one = PagedList[RegisteredModel](
            items=[self.generate_mock_registered_model()], token="token"
        )
        mock_response_data_two = PagedList[RegisteredModel](items=[self.generate_mock_registered_model()], token=None)
        self.client.client = MagicMock()
        self.client.client.search_registered_models.side_effect = [mock_response_data_one, mock_response_data_two]

        response: List[RegisteredModel] = self.client.get_registered_models()
        self.assertEqual(response, [mock_response_data_one[0], mock_response_data_two[0]])

    def test_load_model_by_version(self):
        with patch("mlflow.pyfunc.load_model") as patched_load_model:
            name: str = "mock-name"
            version: str = "1"

            self.client.load_model_by_version(name=name, version=version)

            self.assertEqual(patched_load_model.call_count, 1)
            patched_load_model.assert_called_with(model_uri=f"models:/{name}/{version}")

    def test_load_model_by_stage(self):
        with patch("mlflow.pyfunc.load_model") as patched_load_model:
            name: str = "mock-name"
            stage: str = "MockStage"

            self.client.load_model_by_stage(name=name, stage=stage)

            self.assertEqual(patched_load_model.call_count, 1)
            patched_load_model.assert_called_with(model_uri=f"models:/{name}/{stage}")


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(TestAnacondaMlFlowClient())
