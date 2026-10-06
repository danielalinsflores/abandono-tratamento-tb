from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ANOS = [2022, 2023, 2024]


def main() -> None:
    for ano in ANOS:
        caminho = RAW_DATA_DIR / f"TUBE_{ano}.parquet"
        df = pd.read_parquet(caminho)

        print(f"\n--- {ano} ---")
        print(f"Total de linhas: {len(df)}")
        print(f"Total de colunas: {len(df.columns)}")

        if "SG_UF" in df.columns:
            rj = df[df["SG_UF"] == "33"]
            print(f"Linhas no RJ (SG_UF=33): {len(rj)}")
        else:
            print("Coluna SG_UF não encontrada — colunas disponíveis com 'UF':")
            print([c for c in df.columns if "UF" in c.upper()])

        if "SITUA_ENCE" in df.columns:
            print("Valores únicos em SITUA_ENCE:")
            print(df["SITUA_ENCE"].value_counts(dropna=False))
        else:
            print("Coluna SITUA_ENCE não encontrada — colunas disponíveis com 'ENCE' ou 'SITUA':")
            print([c for c in df.columns if "ENCE" in c.upper() or "SITUA" in c.upper()])

""" TESTE PARA VERIFICAR O TIPO DE DADOS DE UMA COLUNA ESPECÍFICA
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

df = pd.read_parquet(RAW_DATA_DIR / "TUBE_2022.parquet")

print("Tipo da coluna SG_UF:", df["SG_UF"].dtype)
print("Valores únicos de SG_UF (primeiros 10):")
print(df["SG_UF"].unique()[:10])"""


if __name__ == "__main__":
    main()