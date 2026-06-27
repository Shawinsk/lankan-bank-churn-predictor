from pathlib import Path
import os
from urllib.parse import urlparse

import pandas as pd
import joblib
import scipy.sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

from mlFlowProject.entity.config_entity import ModelEvaluationConfig
from mlFlowProject.utils.common import save_json


if os.environ.get("MLFLOW_TRACKING_URI") is not None:
    os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI")
if os.environ.get("MLFLOW_TRACKING_USERNAME") is not None:
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME")
if os.environ.get("MLFLOW_TRACKING_PASSWORD") is not None:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD")



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)

        return accuracy, precision, recall, f1
    

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Load preprocessor and transform features to match model's expected shape
        preprocessor = joblib.load(self.config.preprocessor_path)
        features = ['Age', 'Tenure_Years', 'Account_Balance_LKR', 'District', 'CRIB_Status', 'Recent_Loan_Rejected', 'Digital_Banking_User']
        test_x_transformed = preprocessor.transform(test_x[features])
        if scipy.sparse.issparse(test_x_transformed):
            test_x_transformed = test_x_transformed.toarray()
            
        test_x_df = pd.DataFrame(test_x_transformed, columns=preprocessor.get_feature_names_out())

        if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
            mlflow.set_registry_uri(self.config.mlflow_uri)
        else:
            mlflow.set_registry_uri("")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predictions = model.predict(test_x_df)

            (accuracy, precision, recall, f1) = self.eval_metrics(test_y, predictions)
            
            # Saving metrics as local
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1-score", f1)

            # Model registry does not work with file store
            trusted_types = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier"]
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="LightGBM", skops_trusted_types=trusted_types)
            else:
                mlflow.sklearn.log_model(model, "model", skops_trusted_types=trusted_types)
