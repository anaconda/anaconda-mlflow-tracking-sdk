import shlex
import subprocess
import time
import unittest
from typing import List, Optional

import psutil
import requests
import yaml
from mlflow.entities import Experiment, Run
from requests import Response

from src.anaconda.mlflow.tracking.sdk import AnacondaMlFlowClient, build_mlflow_client


def load_commands() -> dict:
    with open(file="anaconda-project.yml", mode="r", encoding="utf-8") as file:
        project: dict = yaml.safe_load(file)
    return project


class TestClient(unittest.TestCase):
    process: Optional[subprocess.Popen]
    project: Optional[dict]
    client: Optional[AnacondaMlFlowClient]

    def start_server(self):
        shell_out_cmd: str = self.project["commands"]["server"]["unix"]
        args = shlex.split(shell_out_cmd)

        try:
            self.process = subprocess.Popen(args, stdout=subprocess.PIPE)
        except Exception as error:
            # Any failure here is a failed test.
            self.fail(f"Failed to start server process: {str(error)}")

        max_tries: int = 10
        max_wait: int = 5

        current_try: int = 1
        while current_try <= max_tries:
            current_try += 1

            try:
                response: Response = requests.get(url="http://0.0.0.0:5000")

                if response.status_code != 200:
                    print("waiting ...")
                    time.sleep(max_wait)
                else:
                    print("server online")
                    break

            except requests.exceptions.ConnectionError:
                print("waiting ...")
                time.sleep(max_wait)

        self.assertLessEqual(current_try, max_tries)

    def stop_server(self):
        print("shutting down server")
        try:
            if self.process is not None:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        except Exception as error:
            # Any failure here is a failed test.
            self.fail(f"Failed to stop server process: {str(error)}")

    def setUp(self):
        self.project = load_commands()
        self.start_server()
        self.client: AnacondaMlFlowClient = AnacondaMlFlowClient(client=build_mlflow_client())

    def tearDown(self):
        self.stop_server()

    def test_get_experiments(self):
        experiments: List[Experiment] = self.client.get_experiments()
        self.assertEqual(len(experiments), 2)

        default = [experiment for experiment in experiments if experiment.experiment_id == "0"][0]
        self.assertEqual(default.artifact_location, "mlflow-artifacts:/0")

    def test_get_experiment_runs(self):
        runs: List[Run] = self.client.get_experiment_runs(experiment_id="1")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].info.experiment_id, "1")


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(TestClient())
