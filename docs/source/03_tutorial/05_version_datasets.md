# Versioning Kedro DataSets
## What is artifact tracking?

Mlflow defines artifacts as "any data a user may want to track during code execution". This includes, but is not limited to:
- data needed for the model (e.g encoders, vectorizer, the machine learning model itself...)
- graphs (e.g. ROC or PR curve, importance variables, margins,  confusion matrix...)

Artifacts is a very flexible and convenient way to "bind" any data type to your code execution.
Mlflow process for such binding is to :
1. Persist the data locally in the desired file format
2. Upload the data to the [artifact store](./03_configuration.md)

## How to version data in a kedro project?

kedro-mlflow introduces a new ``AbstractDataSet`` called ``MlflowDataSet``. It is a wrapper for any ``AbstractDataSet`` which decorates the underlying dataset ``save`` method and logs the file automatically in mlflow as an artifact each time the ``save`` method is called.

Since it is a ``AbstractDataSet``, it can be used with the YAML API. Assume that you have the following entry in the ``catalog.yml``:

```yaml
my_dataset_to_version:
    type: pandas.CSVDataSet
    filepath: /path/to/a/destination/file.csv
```

You can change it to:

```yaml
my_dataset_to_version:
    type: kedro_mlflow.io.MlflowDataSet
    data_set:
        type: pandas.CSVDataSet  # or any valid kedro DataSet
        filepath: /path/to/a/LOCAL/destination/file.csv # must be a local file, wherever you want to log the data in the end
```
and this dataset will be automatically versioned in each pipeline execution.

## Frequently asked questions
### Can I pass extra parameters to the ``MlflowDataSet`` for finer control?
The ``MlflowDataSet`` takes a ``data_set`` argument which is a python dictionary passed to the ``__init__`` method of the dataset declared in ``type``. It means that you can pass any arguments accepted by the underlying dataset in this dictionary. If you want to pass ``load_args`` and ``save_args`` in the previous example, add them in the ``data_set`` argument:

```yaml
my_dataset_to_version:
    type: kedro_mlflow.io.MlflowDataSet
    data_set:
        type: pandas.CSVDataSet  # or any valid kedro DataSet
        filepath: /path/to/a/local/destination/file.csv
        load_args:
            sep: ;
        save_args:
            sep: ;
        # ... any other valid arguments for data_set
```

### Can I use the ``MlflowDataSet`` in interactive mode?
Like all Kedro ``AbstractDataSet``, ``MlflowDataSet`` is callable in the python API:
```python
from kedro_mlflow.io import MlflowDataSet
from kedro.extras.datasets.pandas import CSVDataSet
csv_dataset = MlflowDataSet(data_set={"type": CSVDataSet, # either a string "pandas.CSVDataSet" or the class
                                      "filepath": r"/path/to/a/local/destination/file.csv"})
csv_dataset.save(data=pd.DataFrame({"a":[1,2], "b": [3,4]}))
```

### How do I upload an artifact to a non local destination (e.g. an S3 or blog storage)?
The location where artifact will be stored does not depends of the logging function but rather on the artifact store specified when configuring the mlflow server. Read mlflow documentation to see:
- how to [configure a mlflow tracking server](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers)
- how to [configure an artifact store](https://www.mlflow.org/docs/latest/tracking.html#id10) with cloud storage.

You can also refer to [this issue](https://github.com/Galileo-Galilei/kedro-mlflow/issues/15) for further details.

In ``kedro-mlflow==0.2.0`` you must configure these elements by yourself. Further releases will introduce helpers for configuration.

### Can I log an artifact in a specific run?
The ``MlflowDataSet`` has an extra argument ``run_id`` which specifies the run in which the artifact will be logged. **Be cautious, because this argument will take precedence over the current run** when you call ``kedro run``, causing the artifact to be logged in another run that all the other data of the run.
```yaml
my_dataset_to_version:
    type: kedro_mlflow.io.MlflowDataSet
    data_set:
        type: pandas.CSVDataSet  # or any valid kedro DataSet
        filepath: /path/to/a/local/destination/file.csv
    run_id: 13245678910111213  # a valid mlflow run to log in. If None, default to active run
```

### Can I create a remote folder/subfolders architecture to organize the artifacts ?
The ``MlflowDataSet`` has an extra argument ``run_id`` which specifies a remote subfolder where the artifact will be logged. It must be a relative path.
```yaml
my_dataset_to_version:
    type: kedro_mlflow.io.MlflowDataSet
    data_set:
        type: pandas.CSVDataSet  # or any valid kedro DataSet
        filepath: /path/to/a/local/destination/file.csv
    artifact_path: reporting  # relative path where the remote artifact must be stored. if None, saved in root folder.
```
