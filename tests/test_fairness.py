import pandas as pd
import numpy as np
import pickle
import pytest
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

DATA_PATH = 'data/processed/df_young_model_ready_2024.csv'
FAIR_PATH = 'data/processed/df_young_fair_2024.csv'
MODEL_PATH = 'models/xgb.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

def test_dpd_range():
    """DPD is between 0 and 1"""
    df = pd.read_csv(DATA_PATH)
    sensitive = pd.read_csv(FAIR_PATH)\
        .loc[df.index, 'sex_of_driver']
    model = pickle.load(open(MODEL_PATH, 'rb'))
    features = pickle.load(
        open(FEATURES_PATH, 'rb'))
    X = df[features]
    y = df['high_risk']
    y_pred = model.predict(X)
    dpd = demographic_parity_difference(
        y, y_pred, 
        sensitive_features=sensitive)
    assert 0 <= abs(dpd) <= 1

def test_eod_range():
    """EOD is between 0 and 1"""
    df = pd.read_csv(DATA_PATH)
    sensitive = pd.read_csv(FAIR_PATH)\
        .loc[df.index, 'sex_of_driver']
    model = pickle.load(open(MODEL_PATH, 'rb'))
    features = pickle.load(
        open(FEATURES_PATH, 'rb'))
    X = df[features]
    y = df['high_risk']
    y_pred = model.predict(X)
    eod = equalized_odds_difference(
        y, y_pred,
        sensitive_features=sensitive)
    assert 0 <= abs(eod) <= 1

def test_perfect_fairness_baseline():
    """When all predictions are equal,
    DPD should be 0"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    sensitive = np.array([1, 1, 2, 2])
    dpd = demographic_parity_difference(
        y_true, y_pred,
        sensitive_features=sensitive)
    assert dpd == 0