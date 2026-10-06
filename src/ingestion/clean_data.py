from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

ANOS = [2022, 2023, 2024]
UF_ALVO = "33"  # RJ, como string - confirmado que SG_UF vem como texto

# Códigos de SITUA_ENCE que definem abandono (confirmado no dicionário oficial do SINAN-TB: Abandono=2, Abandono Primário=10)
CODIGOS_ABANDONO = ["2", "10"]

# Colunasa a usar como ponto de partida - pode-se expandir depois com mais features clínicas/demográficas
COLUNAS_UTEIS = [
    "NU_ANO",
    "SG_UF",
    "ID_MUNICIP",
    "DT_NOTIFIC",
    "CS_SEXO",
    "NU_IDADE_N",
    "CS_ESCOL_N",   # corrigido (era CS_ESCOLAR)
    "FORMA",
    "TRATAMENTO",   # corrigido (era ENTRADA) - tipo de entrada: caso novo, recidiva, reingresso após abandono, etc.
    "AGRAVAIDS",
    "AGRAVALCOO",
    "AGRAVDROGA",   # corrigido (era AGRAVDROGAS)
    "HIV",
    "POP_LIBER",
    "POP_RUA",
    "POP_SAUDE",    # novo - população em situação de rua/vulnerabilidade social relacionada a saúde
    "POP_IMIG",     # novo - população imigrante
    "SITUA_ENCE",
]


def carregar_ano(ano: int) -> pd.DataFrame:
    caminho = RAW_DATA_DIR / f"TUBE_{ano}.parquet"
    df = pd.read_parquet(caminho)
    colunas_existentes = [c for c in COLUNAS_UTEIS if c in df.columns]
    return df[colunas_existentes]


def filtrar_rj(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["SG_UF"] == UF_ALVO].copy()


def criar_target(df: pd.DataFrame) -> pd.DataFrame:
    #Cria a coluna 'abandonou' (1/0) a partir de SITUA_ENCE, e descarta linhas sem desfecho definido (em branco/não encerrado).
    antes = len(df)
    df = df[df["SITUA_ENCE"].notna() & (df["SITUA_ENCE"] != "")].copy()
    print(f"  Descartadas {antes - len(df)} linhas sem desfecho definido.")

    df["abandonou"] = df["SITUA_ENCE"].isin(CODIGOS_ABANDONO).astype(int)
    return df


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    partes = []

    for ano in ANOS:
        print(f"Processando ano {ano}...")
        df = carregar_ano(ano)
        print(f"  Total bruto: {len(df)} linhas")

        df = filtrar_rj(df)
        print(f"  Após filtro RJ: {len(df)} linhas")

        df = criar_target(df)
        taxa = df["abandonou"].mean() * 100
        print(f"  Taxa de abandono: {taxa:.1f}%")

        partes.append(df)

    df_final = pd.concat(partes, ignore_index=True)
    print(f"\nTotal consolidado (2022-2024, RJ): {len(df_final)} linhas")
    print(f"Taxa de abandono geral: {df_final['abandonou'].mean() * 100:.1f}%")

    destino = PROCESSED_DATA_DIR / "tb_rj_consolidado.csv"
    df_final.to_csv(destino, index=False)
    print(f"Salvo em {destino}")


if __name__ == "__main__":
    main()