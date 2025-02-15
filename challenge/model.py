import logging
from pathlib import Path
from typing import List, Tuple, Union

import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import xgboost as xgb


class DelayModel:
    
    DATA_SPLITTING_RANDOM_STATE = 42
    TEST_SIZE = 0.33
    MODEL_RANDOM_STATE = 1
    LEARNING_RATE = 0.01
    DELAY_THRESHOLD_MINUTES = 15
    MODEL_FILE_NAME = Path("model.json")
    MODEL_PATH = Path("models")

    def __init__(self, _model: Union[BaseEstimator, ClassifierMixin] = None, target_column: str = ""):
        self._model = _model or self._load_model(self.complete_model_path)
        self.target_column = target_column
        self._model = _model or self._load_model(self.complete_model_path)
        self.xgb_model = xgb.Booster()
        path_to_file = "../models/model.json"

        try:
            self.xgb_model.load_model(path_to_file)
        except xgb.core.XGBoostError as e:
            logging.error(f"Error loading XGBoost model: {str(e)}")
            self.xgb_model = None

    @property
    def complete_model_path(self) -> Path:
        return self.MODEL_PATH / self.MODEL_FILE_NAME

    def _get_minute_diff(self, data: pd.DataFrame) -> pd.Series:
        try:
            fecha_o = pd.to_datetime(data["Fecha-O"])
            fecha_i = pd.to_datetime(data["Fecha-I"])
            return (fecha_o - fecha_i).dt.total_seconds() / 60
        except (ValueError, KeyError) as e:
            raise ValueError("Invalid input data or date format") from e

    def _create_one_hot_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

    def _create_delay_target(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        data["min_diff"] = self._get_minute_diff(data)
        delay_target = np.where(data["min_diff"] > self.DELAY_THRESHOLD_MINUTES, 1, 0)
        return pd.DataFrame({target_column: delay_target}, index=data.index)

    def _load_model(self, model_path: Path) -> Union[BaseEstimator, ClassifierMixin]:
        logging.info(f"Loading model from {model_path}")
        try:
            if not model_path.is_file():
                raise FileNotFoundError(f"No file found at {model_path}")
            if model_path.suffix.lower() != ".json":
                logging.warning(f"File {model_path} does not have a .json extension. Attempting to load anyway.")
            model = XGBClassifier()
            model.load_model(str(model_path))
            logging.info("Model loaded successfully")
            return model
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            logging.error(f"Error loading model: {str(e)}")

    def _calculate_class_weights(self, target: pd.Series) -> dict:
        classes = np.unique(target)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=target)
        class_weights = dict(zip(classes, weights))
        logging.info(f"Class weights: {class_weights}")
        return class_weights

    def top_features(self, n: int = 10) -> List[str]:
        if self._model is None:
            logging.warning("Model wasn't found.")
            self._load_model(self.complete_model_path)
        return sorted(self._model.get_booster().get_fscore().items(), key=lambda x: x[1], reverse=True)[:n]

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        target = self._create_delay_target(data, target_column) if target_column else None
        features = self._create_one_hot_features(data).reindex(columns=self.top_features(), fill_value=0)
        return (features, target) if target_column else features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=self.TEST_SIZE, random_state=self.DATA_SPLITTING_RANDOM_STATE)
        class_weights = self._calculate_class_weights(target[self.target_column])
        scale_pos_weight = class_weights[1] / class_weights[0]
        model = XGBClassifier(random_state=self.MODEL_RANDOM_STATE, learning_rate=self.LEARNING_RATE, scale_pos_weight=scale_pos_weight)
        model.fit(x_train, y_train)
        logging.info("Finished training, calculating test metrics...")
        y_pred = model.predict(x_test)
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        self.complete_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(self.complete_model_path)
        self._model = model

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            logging.warning("Model wasn't found.")
            self._load_model(self.complete_model_path)
        return self._model.predict(features).tolist()
