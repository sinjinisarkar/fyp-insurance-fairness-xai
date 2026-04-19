import pandas as pd
import pytest

DATA_PATH = 'data/processed/df_young_model_ready_2024.csv'
FAIR_PATH = 'data/processed/df_young_fair_2024.csv'

def test_age_filter():
    """Age filter correctly restricts 
    to drivers aged 17-25"""
    df = pd.read_csv(FAIR_PATH)
    assert df['age_of_driver'].min() >= 17
    assert df['age_of_driver'].max() <= 25

def test_sex_codes():
    """Only valid sex codes retained"""
    df = pd.read_csv(FAIR_PATH)
    assert set(df['sex_of_driver'].unique())\
        .issubset({1, 2})

def test_binary_target():
    """Target variable contains 
    only 0 and 1"""
    df = pd.read_csv(DATA_PATH)
    assert set(df['high_risk'].unique()) == {0, 1}

def test_no_missing_values():
    """Feature matrix has no 
    missing values"""
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0

def test_protected_attributes_excluded():
    """Protected attributes present in saved 
    dataset but excluded from model training 
    feature matrix X"""
    df = pd.read_csv(DATA_PATH)
    
    # They should be IN the saved dataset
    # (separated at training time into matrix A)
    assert 'sex_of_driver' in df.columns
    assert 'age_band_of_driver' in df.columns
    
    # But the model feature matrix should NOT
    # include them - verified by checking
    # feature_names.pkl
    import pickle
    features = pickle.load(
        open('models/feature_names.pkl', 'rb'))
    assert 'sex_of_driver' not in features
    assert 'age_band_of_driver' not in features