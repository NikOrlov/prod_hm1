from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from entities.train_pipeline_params import TrainingPipelineParams, FeatureParams
import numpy as np
import pandas as pd


def numerical_pipeline() -> Pipeline:
    pipeline = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                         ('normalize', StandardScaler())])
    return pipeline


def categorical_pipeline() -> Pipeline:
    pipeline = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))])
    return pipeline


def build_pipeline(features: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer([('numerical_part', numerical_pipeline(), features.feature_params.numerical_features),
                                     ('categorical_part', categorical_pipeline(), features.feature_params.categorical_features)])
    return transformer


def make_features(data: pd.DataFrame, features: FeatureParams) -> np.ndarray:
    transformer = build_pipeline(features)
    return transformer.fit_transform(data)


def extract_target(data: pd.DataFrame, target: FeatureParams) -> pd.Series:
    return data[target.target_feature]


if __name__ == '__main__':
    from data.make_dataset import read_data
    from entities.train_pipeline_params import read_training_pipeline_params
    path_data = '../data/heart_nan.csv'
    path_config = '../config/train_config.yaml'
    params = read_training_pipeline_params(path_config)
    dataset = read_data(path_data).drop(params.feature_params.target_feature, axis=1)
    data_transformed = make_features(dataset, params)
    assert data_transformed.shape == dataset.shape
