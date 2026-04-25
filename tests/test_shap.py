import joblib
import numpy as np
import pandas as pd
import pytest
import shap

MODEL_PATH = 'models/xgb.pkl'
FEATURES_PATH = 'models/feature_names.pkl'
DATA_PATH = 'data/processed/df_young_model_ready_2024.csv'

def test_shap_values_shape():
    """SHAP values have correct shape
    matching test input"""
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[features].head(10)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    values = np.array(shap_values)
    if len(values.shape) == 3:
        values = values[:, :, 1]
    assert values.shape == (10, len(features))

def test_shap_dimensionality_fix():
    """3D SHAP array correctly reduced
    to 2D for binary classification"""
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[features].head(5)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    values = np.array(shap_values)
    if len(values.shape) == 3:
        values = values[:, :, 1]
    assert len(values.shape) == 2