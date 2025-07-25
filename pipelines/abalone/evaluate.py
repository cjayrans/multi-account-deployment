"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile
import os
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_squared_error, root_mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    try:
        logger.info("=== EVALUATION SCRIPT START ===")
        logger.debug("Starting evaluation.")
        # 1) List the model & test inputs we actually received:
        model_dir = "/opt/ml/processing/model"
        test_dir = "/opt/ml/processing/test"
        logger.info("Model dir contents: %s", os.listdir(model_dir))
        logger.info("Test  dir contents: %s", os.listdir(test_dir))

        # 2) Extract the tarball
        model_path = "/opt/ml/processing/model/model.tar.gz"
        with tarfile.open(model_path) as tar:
            tar.extractall(path=".")
        logger.info("Extracted tarball successfully.")

        # 3) Load model
        logger.info("Loading xgboost model pickle...")
        logger.debug("Loading xgboost model.")
        model = pickle.load(open("xgboost-model", "rb"))
        logger.info("Model loaded: %s", model)

        # 4) Read test data
        logger.debug("Reading test data.")
        test_path = "/opt/ml/processing/test/test.csv"
        df = pd.read_csv(test_path, header=None)
        logger.debug("Reading test data.")
        y_test = df.iloc[:, 0].to_numpy()
        df.drop(df.columns[0], axis=1, inplace=True)
        X_test = xgboost.DMatrix(df.values)

        # 5) Predict
        logger.info("Performing predictions against test data.")
        predictions = model.predict(X_test)

        # 6) Compute RMSE
        logger.debug("Calculating root mean squared error.")
        rmse = root_mean_squared_error(y_test, predictions)
        std = np.std(y_test - predictions)
        logger.info("Computed RMSE = %f, STD = %f", rmse, std)
        report_dict = {
            "regression_metrics": {
                "rmse": {
                    "value": rmse,
                    "standard_deviation": std
                },
            },
        }

        output_dir = "/opt/ml/processing/evaluation"
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        eval_path = os.path.join(output_dir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(report_dict, f)
            f.flush()
            os.fsync(f.fileno())
        logger.info("Wrote evaluation report to %s", eval_path)

        # 8) Confirm it really is there
        if os.path.exists(eval_path):
            logger.info("evaluation.json exists")
        else:
            logger.error("evaluation.json does NOT exist")

    except Exception as e:
        logger.exception("Unexpected error in evaluation script")
        # Exit non-zero so the processing step fails visibly
        raise

        # logger.info("Writing out evaluation report with rmse: %f", rmse)
        # evaluation_path = f"{output_dir}/evaluation.json"
        # with open(evaluation_path, "w") as f:
        #     f.write(json.dumps(report_dict))
