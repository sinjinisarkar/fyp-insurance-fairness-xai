import pandas as pd
import pytest

DATA_PATH = 'data/processed/df_young_model_ready_2024.csv'

LEAKAGE_COLS = [
    'collision_severity',
    'number_of_casualties',
    'collision_injury_based',
    'did_police_officer_attend_scene_of_accident',
    'enhanced_severity_collision',
    'collision_adjusted_severity_serious',
    'collision_adjusted_severity_slight'
]

def test_no_leakage_columns():
    """All leakage columns removed 
    from feature matrix"""
    df = pd.read_csv(DATA_PATH)
    for col in LEAKAGE_COLS:
        assert col not in df.columns, \
            f"Leakage column still present: {col}"

def test_no_identifier_columns():
    """Identifier columns removed 
    from feature matrix"""
    df = pd.read_csv(DATA_PATH)
    id_cols = [
        'collision_index',
        'collision_ref_no',
        'vehicle_reference'
    ]
    for col in id_cols:
        assert col not in df.columns, \
            f"Identifier column still present: {col}"