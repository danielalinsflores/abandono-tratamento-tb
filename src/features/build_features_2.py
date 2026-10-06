from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
df = pd.read_parquet(RAW_DATA_DIR / "TUBE_2022.parquet")

termos = ["ESCOL", "ENTRADA", "DROGA", "TRATAMENTO", "POP_"]
for termo in termos:
    encontradas = [c for c in df.columns if termo in c.upper()]
    print(f"{termo}: {encontradas}")