import importlib
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from kedro.io import AbstractVersionedDataSet, Version
from kedro.io.core import DataSetError
from mlflow.tracking import MlflowClient


class MlflowModelDataSet(AbstractVersionedDataSet):
    """Wrapper for saving, logging and loading for all MLflow model flavor."""

    def __init__(
        self,
        flavor: str,
        path: str,
        run_id: Optional[str] = None,
        load_args: Dict[str, Any] = {},
        save_args: Dict[str, Any] = {},
        log_args: Dict[str, Any] = {},
        version: Version = None,
    ) -> None:
        """Intialize the Kedro MlflowModelDataSet.

        Parameters are passed from the Data Catalog.

        During save, the model is first saved locally at `path` and then
        logged to MLflow.
        During load, the model is either pulled from MLflow run with `run_id`
        or loaded from the local `path`.

        Args:
            flavor (str): Built-in or custom MLflow model flavor module.
                Must be Python-importable.
            path (str): Path to store the dataset locally.
            run_id (Optional[str], optional): MLflow run ID to use to load
                the model from. Defaults to None.
            load_args (Dict[str, Any], optional): Arguments to `load_model`
                function from specified `flavor`. Defaults to {}.
            save_args (Dict[str, Any], optional): Arguments to `save_model`
                function from specified `flavor`. Defaults to {}.
            log_args (Dict[str, Any], optional): Arguments to `log_model`
                function from specified `flavor`. Defaults to {}.
            version (Version, optional): Kedro version to use. Defaults to None.

        Raises:
            DataSetError: When passed `flavor` does not exist.
        """
        super().__init__(PurePosixPath(path), version)
        self._flavor = flavor
        self._path = path
        self._run_id = run_id
        self._load_args = load_args
        self._save_args = save_args
        self._log_args = log_args
        self._version = version

        mlflow_model_module_exists = importlib.util.find_spec(self._flavor)

        if not mlflow_model_module_exists:
            raise DataSetError(f"{self._flavor} not found")

        self._mlflow_model_module = importlib.import_module(self._flavor)

    def _load(self) -> Any:
        """Loads an MLflow model from local path or from MLflow run.

        Returns:
            Any: Deserialized model.
        """
        # If `run_id` is specified, pull the model from MLflow.
        if self._run_id:
            mlflow_client = MlflowClient()
            run = mlflow_client.get_run(self._run_id)
            load_path = f"{run.info.artifact_uri}/{Path(self._path).name}"
        # Alternatively, use local path to load the model.
        else:
            load_path = str(self._get_load_path())
        return self._mlflow_model_module.load_model(load_path, **self._load_args)

    def _save(self, model: Any) -> None:
        """Save a model to local path and then logs it to MLflow.

        Args:
            model (Any): A model object supported by the given MLflow flavor.
        """
        save_path = self._get_save_path()
        # In case of an unversioned model we need to remove the save path
        # because MLflow cannot overwrite the target directory.
        if Path(save_path).exists():
            shutil.rmtree(save_path)
        self._mlflow_model_module.save_model(model, save_path, **self._save_args)
        self._mlflow_model_module.log_model(model, save_path.name, **self._log_args)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            flavor=self._flavor,
            path=self._path,
            run_id=self._run_id,
            load_args=self._load_args,
            save_args=self._save_args,
            log_args=self._log_args,
            version=self._version,
        )
