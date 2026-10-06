import pysus

df = pysus.ftp.list_files("SINAN", source="origin")
print(f"Total de arquivos: {len(df)}")
print(df.columns.tolist())
if len(df) > 0:
    print("Grupos disponíveis:")
    print(df["group"].unique())