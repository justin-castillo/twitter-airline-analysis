# tests/test_data_prep.py
from twitter_airline_analysis import data_prep as dp


def test_clean_text():
    raw = "Check this! https://t.co/xyz @United #delay"
    cleaned = dp.clean_text(raw)
    assert cleaned == "check this!"


def test_preprocess_shape():
    df_raw = dp.load_raw()
    df_tidy = dp.preprocess(df_raw)
    assert len(df_tidy) == len(df_raw)
    assert "clean_text" in df_tidy.columns
