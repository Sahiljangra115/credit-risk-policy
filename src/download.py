"""Download the seven Home Credit tables (parquet) from the Hugging Face mirror into data/raw/. About 1.1 GB."""
from huggingface_hub import snapshot_download
from . import config as C

REPO = "mohameddhameem/home-credit-default-risk"
FOLDERS = ["application_train_dated", "bureau", "bureau_balance", "previous_application",
           "installments_payments", "POS_CASH_balance", "credit_card_balance"]

if __name__ == "__main__":
    snapshot_download(REPO, repo_type="dataset", local_dir=C.RAW, allow_patterns=[f"{f}/*.parquet" for f in FOLDERS])
    print("downloaded to", C.RAW)
