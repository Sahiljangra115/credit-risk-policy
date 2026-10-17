"""Run the committed SQL on a tiny hand-made dataset and check exact values."""
import duckdb
import pandas as pd
import pytest
from src import features


@pytest.fixture()
def con():
    c = duckdb.connect()
    app = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "application_date": ["2020-01-15", "2020-02-10", "2020-02-20"],
        "CODE_GENDER": ["F", "M", "F"], "REGION_RATING_CLIENT": [2, 2, 3],
        "AMT_CREDIT": [1000.0, 2000.0, 500.0], "AMT_INCOME_TOTAL": [500.0, 1000.0, 0.0], "AMT_ANNUITY": [100.0, 200.0, 50.0],
        "AMT_GOODS_PRICE": [900.0, 2000.0, 500.0], "CNT_CHILDREN": [0, 1, 2], "CNT_FAM_MEMBERS": [2, 3, 4],
        "NAME_CONTRACT_TYPE": ["Cash loans"] * 3, "NAME_INCOME_TYPE": ["Working"] * 3, "NAME_EDUCATION_TYPE": ["Secondary"] * 3,
        "NAME_FAMILY_STATUS": ["Married"] * 3, "NAME_HOUSING_TYPE": ["House / apartment"] * 3, "FLAG_OWN_CAR": ["Y"] * 3,
        "FLAG_OWN_REALTY": ["Y"] * 3, "OCCUPATION_TYPE": ["Laborers", None, "Managers"],
        "EXT_SOURCE_1": [0.2, None, 0.6], "EXT_SOURCE_2": [0.4, 0.5, 0.6], "EXT_SOURCE_3": [None, None, None],
        "DAYS_BIRTH": [-365 * 40, -365 * 30, -365 * 50], "DAYS_EMPLOYED": [-730, 365243, -365],
    })
    bureau = pd.DataFrame({"SK_ID_CURR": [1, 1], "SK_ID_BUREAU": [10, 11], "CREDIT_ACTIVE": ["Active", "Closed"],
                           "CREDIT_TYPE": ["Consumer credit", "Car loan"], "AMT_CREDIT_SUM": [100.0, 300.0],
                           "AMT_CREDIT_SUM_DEBT": [50.0, 0.0], "AMT_CREDIT_SUM_OVERDUE": [5.0, 0.0], "CREDIT_DAY_OVERDUE": [7, 0]})
    bb = pd.DataFrame({"SK_ID_BUREAU": [10, 10, 10, 10, 11], "MONTHS_BALANCE": [0, -1, -2, -3, 0], "STATUS": ["0", "2", "C", "0", "0"]})
    prev = pd.DataFrame({"SK_ID_PREV": [1, 2, 3], "SK_ID_CURR": [1, 1, 2], "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
                         "AMT_APPLICATION": [100.0, 200.0, 50.0], "AMT_CREDIT": [100.0, 0.0, 50.0], "DAYS_DECISION": [-100, -50, -10]})
    inst = pd.DataFrame({"SK_ID_PREV": [1, 1, 3], "SK_ID_CURR": [1, 1, 2], "DAYS_INSTALMENT": [-100, -50, -20],
                         "DAYS_ENTRY_PAYMENT": [-95, -50, -25], "AMT_INSTALMENT": [10.0, 10.0, 10.0], "AMT_PAYMENT": [10.0, 5.0, 10.0]})
    pos = pd.DataFrame({"SK_ID_PREV": [1, 1], "SK_ID_CURR": [1, 1], "MONTHS_BALANCE": [-1, -2], "CNT_INSTALMENT": [6.0, 6.0],
                        "NAME_CONTRACT_STATUS": ["Active", "Completed"], "SK_DPD": [0, 10]})
    cc = pd.DataFrame({"SK_ID_PREV": [9], "SK_ID_CURR": [2], "AMT_CREDIT_LIMIT_ACTUAL": [1000.0], "AMT_BALANCE": [250.0],
                       "AMT_DRAWINGS_CURRENT": [20.0], "SK_DPD": [0]})
    for name, d in [("application", app), ("bureau", bureau), ("bureau_balance", bb), ("previous_application", prev),
                    ("installments_payments", inst), ("pos_cash_balance", pos), ("credit_card_balance", cc)]:
        c.register(f"_{name}", d)
        c.execute(f"CREATE VIEW {name} AS SELECT * FROM _{name}")
    return c


def test_model_table_one_row_per_applicant_and_exact_values(con):
    df = features.model_table(con).set_index("SK_ID_CURR")
    assert list(df.index.sort_values()) == [1, 2, 3] and df.index.is_unique
    assert df.loc[1, "credit_income_ratio"] == pytest.approx(2.0)
    assert pd.isna(df.loc[3, "credit_income_ratio"])                       # zero income -> NULL, not inf
    assert df.loc[1, "age_years"] == pytest.approx(40 * 365 / 365.25)
    assert df.loc[2, "employed_flag_missing"] == 1 and pd.isna(df.loc[2, "employed_years"])  # 365243 sentinel
    assert df.loc[1, "ext_source_mean"] == pytest.approx(0.3) and df.loc[3, "ext_source_mean"] == pytest.approx(0.6)
    assert df.loc[1, "app_month"] == "2020-01"


def test_bureau_features(con):
    df = features.model_table(con).set_index("SK_ID_CURR")
    assert df.loc[1, "bureau_n"] == 2 and df.loc[1, "bureau_active_n"] == 1 and df.loc[1, "bureau_type_n"] == 2
    assert df.loc[1, "bureau_debt_ratio"] == pytest.approx(50 / 400)
    assert df.loc[1, "bureau_max_day_overdue"] == 7
    assert df.loc[1, "bureau_bb_worst_status"] == 2
    assert df.loc[1, "bureau_bb_late_share"] == pytest.approx((0.25 + 0.0) / 2)   # loan 10: 1 of 4 late; loan 11: 0 of 1
    assert pd.isna(df.loc[2, "bureau_n"])                                       # no bureau history stays NULL


def test_previous_and_installment_features(con):
    df = features.model_table(con).set_index("SK_ID_CURR")
    assert df.loc[1, "prev_n"] == 2 and df.loc[1, "prev_refused_rate"] == pytest.approx(0.5)
    assert df.loc[1, "inst_days_late_max"] == 5 and df.loc[1, "inst_late_share"] == pytest.approx(0.5)
    assert df.loc[1, "inst_underpaid_n"] == 1
    assert df.loc[2, "inst_late_share"] == pytest.approx(0.0)                   # paid 5 days early
    assert df.loc[1, "inst_pay_ratio_mean"] == pytest.approx(0.75)
    assert df.loc[1, "pos_dpd_share"] == pytest.approx(0.5) and df.loc[1, "pos_dpd_max"] == 10
    assert df.loc[2, "cc_utilisation_mean"] == pytest.approx(0.25)
