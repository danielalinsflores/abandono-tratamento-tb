from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Colunas binárias do tipo "1=Sim / 2=Não / 9=Ignorado" (ou vazio).
# Simplificação adotada: só 1 (Sim) vira 1; tudo o mais (Não, Ignorado,
# ausente) vira 0. Isso é uma decisão de modelagem documentada como
# limitação no README - não distinguimos "não" de "não informado".
COLUNAS_BINARIAS = [
    "AGRAVAIDS", "AGRAVALCOO", "AGRAVDROGA", "HIV",
    "POP_LIBER", "POP_RUA", "POP_SAUDE", "POP_IMIG",
]


def decodificar_idade(valor):
    if pd.isna(valor):
        return None
    valor = int(valor)
    unidade = valor // 1000
    quantidade = valor % 1000
    if unidade == 4:
        return round(quantidade, 1)
    elif unidade == 3:
        return round(quantidade / 12, 1)
    elif unidade == 2:
        return round(quantidade / 365, 1)
    elif unidade == 1:
        return round(quantidade / 8760, 1)
    return None


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "tb_rj_consolidado.csv")

    # Idade
    df["idade_anos"] = df["NU_IDADE_N"].apply(decodificar_idade)
    df = df.drop(columns=["NU_IDADE_N"])

    # Colunas binárias de agravo/vulnerabilidade
    for col in COLUNAS_BINARIAS:
        df[col] = (df[col].astype(str).str.strip() == "1.0").astype(int)

    # Variáveis categóricas -> one-hot encoding
    categoricas = ["CS_SEXO", "CS_ESCOL_N", "FORMA", "TRATAMENTO"]
    df = pd.get_dummies(df, columns=categoricas, dummy_na=True)

    # Imputação simples de idade ausente com a mediana
    df["idade_anos"] = df["idade_anos"].fillna(df["idade_anos"].median())

    # Remove colunas que não entram no modelo
    df_final = df.drop(columns=["SG_UF", "ID_MUNICIP", "DT_NOTIFIC", "SITUA_ENCE"])

    print(f"Shape final: {df_final.shape}")
    print(f"Colunas finais: {df_final.columns.tolist()}")

    destino = PROCESSED_DATA_DIR / "tb_rj_features.csv"
    df_final.to_csv(destino, index=False)
    print(f"Salvo em {destino}")


if __name__ == "__main__":
    main()