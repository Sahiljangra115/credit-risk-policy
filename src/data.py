"""Open the raw tables as DuckDB views. Real files by default; the SYNTHETIC set only when asked for explicitly."""
import duckdb
from . import config as C

# view name -> accepted raw names (csv/parquet file or parquet folder), case-insensitive
TABLES = {
    "application": ["application_train", "application_train_dated"],
    "bureau": ["bureau"],
    "bureau_balance": ["bureau_balance"],
    "previous_application": ["previous_application"],
    "installments_payments": ["installments_payments"],
    "pos_cash_balance": ["pos_cash_balance"],
    "credit_card_balance": ["credit_card_balance"],
}


def _find(root, names):
    if not root.exists():
        return None
    for p in sorted(root.iterdir()):
        stem = p.stem.lower() if p.is_file() else p.name.lower()
        if stem in names:
            if p.is_dir() and list(p.glob("*.parquet")):
                return f"read_parquet('{p}/*.parquet')"
            if p.suffix == ".parquet":
                return f"read_parquet('{p}')"
            if p.suffix == ".csv":
                return f"read_csv('{p}', header=true, sample_size=-1)"
    return None


def connect(force_synthetic=False):
    """Return (con, source) where source is 'REAL' or 'SYNTHETIC'."""
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'; PRAGMA threads=4")
    if force_synthetic:
        from .synth import generate
        generate()
        srcs = {v: _find(C.SYN, n) for v, n in TABLES.items()}
        source = "SYNTHETIC"
    else:
        srcs = {v: _find(C.RAW, n) for v, n in TABLES.items()}
        missing = [v for v, s in srcs.items() if s is None]
        if missing:
            raise FileNotFoundError(f"missing raw tables {missing} in {C.RAW}. Run `python -m src.download`, "
                                    "or `python -m src.run --synthetic` for a fake-data plumbing check.")
        source = "REAL"
    for v, s in srcs.items():
        con.execute(f"CREATE VIEW {v} AS SELECT * FROM {s}")
    return con, source
