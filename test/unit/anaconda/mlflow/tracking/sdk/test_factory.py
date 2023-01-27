import unittest

from mlflow.tracking.client import MlflowClient

from src.anaconda.mlflow.tracking.sdk import build_mlflow_client


class TestFactory(unittest.TestCase):
    def test_build_mlflow_client(self):
        client: MlflowClient = build_mlflow_client()
        self.assertIsInstance(obj=client, cls=MlflowClient)


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(TestFactory())
