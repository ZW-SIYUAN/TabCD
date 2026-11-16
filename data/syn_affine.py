"""Generate all 231 integer rows satisfying x3 = x1 + 2*x2."""

from pathlib import Path

import pandas as pd


def prepare(directory: Path) -> pd.DataFrame:
    frame = (
        pd.DataFrame(
            [(x1, x2, x1 + 2 * x2) for x1 in range(21) for x2 in range(11)],
            columns=["x1", "x2", "x3"],
        )
        .sample(frac=1, random_state=2207)
        .reset_index(drop=True)
    )
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "syn-affine.csv", index=False)
    return frame


if __name__ == "__main__":
    prepare(Path("outputs/data"))
