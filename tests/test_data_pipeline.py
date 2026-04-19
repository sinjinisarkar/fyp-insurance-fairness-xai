import pandas as pd
import pytest

DATA_PATH = 'data/processed/df_young_model_ready_2024.csv'
FAIR_PATH = 'data/processed/df_young_fair_2024.csv'
RAW_MERGE_PATH = 'data/processed/df_young_drivers_2024.csv'

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
    """Protected attributes present in 
    dataset but excluded from model 
    feature matrix"""
    import joblib
    df = pd.read_csv(DATA_PATH)
    assert 'sex_of_driver' in df.columns
    assert 'age_band_of_driver' in df.columns
    features = joblib.load(
        'models/feature_names.pkl')
    assert 'sex_of_driver' not in features
    assert 'age_band_of_driver' not in features

def test_no_duplicates():
    """No duplicate records in 
    merged dataset"""
    df = pd.read_csv(RAW_MERGE_PATH)
    assert df.duplicated().sum() == 0

def test_final_observation_count():
    """Final cleaned dataset contains 
    correct number of observations"""
    df = pd.read_csv(FAIR_PATH)
    assert len(df) >= 13375

def test_sex_codes_after_cleaning():
    """After cleaning sex_of_driver 
    contains only 1 and 2"""
    df = pd.read_csv(FAIR_PATH)
    valid_codes = {1, 2}
    actual_codes = set(
        df['sex_of_driver'].unique())
    assert actual_codes.issubset(valid_codes), \
        f"Invalid sex codes found: {actual_codes - valid_codes}"