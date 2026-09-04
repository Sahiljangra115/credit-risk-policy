"""Run the committed SQL files in DuckDB and build the one-row-per-applicant model table."""
import pandas as pd
from . import config as C

FEATURE_VIEWS = {
    "feat_application": "01_application.sql",
    "feat_bureau": "02_bureau.sql",
    "feat_prev": "03_previous_application.sql",
    "feat_installments": "04_installments.sql",
    "feat_pos": "05_pos_cash.sql",
    "feat_cc": "06_credit_card.sql",
}

APP_BASE = """SELECT *, strftime(CAST(application_date AS DATE), '%Y-%m') AS app_month FROM application"""


def build_views(con):
    """Create app_base and feat_* views on a connection that already has the 7 raw views."""
    con.execute(f"CREATE OR REPLACE VIEW app_base AS {APP_BASE}")
    for view, fn in FEATURE_VIEWS.items():
        sql = (C.SQL / fn).read_text().strip().rstrip(";")
        con.execute(f"CREATE OR REPLACE TABLE {view} AS {sql}")


def model_table(con) -> pd.DataFrame:
    build_views(con)
    sql = (C.SQL / "99_model_table.sql").read_text().strip().rstrip(";")
    return con.execute(sql).df()
