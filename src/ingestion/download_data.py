from pathlib import Path

import pandas as pd
from pysus import PySUS

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ANOS = [2022, 2023, 2024]


def baixar_ano(client: PySUS, ano: int) -> pd.DataFrame:
    """Baixa os dados de TB de um ano.

    A PySUS tem um bug conhecido: alguns arquivos do catálogo têm um campo
    interno (group_id) nulo, o que faz a busca filtrada por grupo descartar
    arquivos que na verdade existem e têm dados válidos no servidor.
    Por isso, buscamos sem filtro de grupo e filtramos manualmente pelo
    nome do arquivo (todo arquivo de TB começa com "TUBE").
    """
    todos = client.query(dataset="sinan", year=ano)
    arquivos = [f for f in todos if "TUBE" in str(f).upper()]

    if not arquivos:
        raise RuntimeError(f"Nenhum arquivo de TB encontrado para {ano}.")

    partes = []
    for arquivo in arquivos:
        try:
            partes.append(pd.read_parquet(arquivo))
        except FileNotFoundError:
            # Caso o arquivo ainda não tenha sido baixado localmente
            local = client.download(arquivo)
            partes.append(pd.read_parquet(local))

    return pd.concat(partes, ignore_index=True)


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with PySUS() as client:
        for ano in ANOS:
            destino = RAW_DATA_DIR / f"TUBE_{ano}.parquet"
            if destino.exists():
                print(f"{destino.name} já existe, pulando.")
                continue

            print(f"Baixando TB {ano}...")
            df = baixar_ano(client, ano)
            df.to_parquet(destino, index=False)
            print(f"Salvo em {destino} ({len(df)} linhas)")


if __name__ == "__main__":
    main()