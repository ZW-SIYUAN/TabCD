"""Download UCI Computer Hardware (DOI: 10.24432/C5830D)."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/29/computer+hardware.zip"
COLUMNS = [
    "VendorName",
    "ModelName",
    "MYCT",
    "MMIN",
    "MMAX",
    "CACH",
    "CHMIN",
    "CHMAX",
    "PRP",
    "ERP",
]


def prepare(directory: Path) -> pd.DataFrame:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "computer-hardware.csv"
    if not target.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open("machine.data") as source:
            frame = pd.read_csv(source, header=None, names=COLUMNS)
        if frame.shape != (209, 10) or frame.isna().any().any():
            raise ValueError("Unexpected UCI Computer Hardware schema")
        frame.to_csv(target, index=False)
    frame = pd.read_csv(target, dtype={"VendorName": str, "ModelName": str})
    if frame.shape != (209, 10) or list(frame.columns) != COLUMNS or frame.isna().any().any():
        raise ValueError("Invalid cached Computer Hardware table")
    return frame.sample(frac=1, random_state=2207).reset_index(drop=True)


if __name__ == "__main__":
    prepare(Path("outputs/data"))
