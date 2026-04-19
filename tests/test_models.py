import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATHS = {
    'lr': 'models/lr_balanced.pkl',
    'rf': 'models/rf_balanced.pkl',
    'xgb': 'models/xgb.pkl'
}
FEATURES_PATH = 'models/feature_names.pkl'

def test_models_load():
    """All three models load 
    without errors"""
    for name, path in MODEL_PATHS.items():
        model = joblib.load(path)
        assert model is not None, \
            f"Model {name} failed to load"

def test_prediction_shape():
    """Models produce correct 
    output shape"""
    features = joblib.load(FEATURES_PATH)
    dummy = pd.DataFrame(
        np.zeros((1, len(features))),
        columns=features)
    for name, path in MODEL_PATHS.items():
        model = joblib.load(path)
        pred = model.predict_proba(dummy)
        assert pred.shape[1] == 2, \
            f"{name} wrong output shape"

def test_probability_range():
    """Model probabilities are 
    between 0 and 1"""
    features = joblib.load(FEATURES_PATH)
    dummy = pd.DataFrame(
        np.zeros((1, len(features))),
        columns=features)
    for name, path in MODEL_PATHS.items():
        model = joblib.load(path)
        pred = model.predict_proba(dummy)
        assert pred[0].min() >= 0
        assert pred[0].max() <= 1

def test_binary_prediction():
    """Models produce binary 
    predictions (0 or 1)"""
    features = joblib.load(FEATURES_PATH)
    dummy = pd.DataFrame(
        np.zeros((1, len(features))),
        columns=features)
    for name, path in MODEL_PATHS.items():
        model = joblib.load(path)
        pred = model.predict(dummy)
        assert pred[0] in {0, 1}