from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DASHBOARDS_DIR = Path(__file__).resolve().parents[2] / "dashboards"


def main() -> None:
    DASHBOARDS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_DIR / "tb_rj_features.csv")
    X = df.drop(columns=["abandonou", "NU_ANO"])
    y = df["abandonou"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = joblib.load(MODELS_DIR / "modelo_regressao_logistica.pkl")
    with open(MODELS_DIR / "threshold.txt") as f:
        threshold = float(f.read())

    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # --- Gráfico 1: Matriz de confusão ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Não abandonou", "Abandonou"],
        yticklabels=["Não abandonou", "Abandonou"],
        ax=ax,
    )
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — Regressão Logística (threshold={threshold})")
    fig.tight_layout()
    fig.savefig(DASHBOARDS_DIR / "matriz_confusao.png", dpi=150)
    plt.close(fig)
    print("Salvo: matriz_confusao.png")

    # --- Gráfico 2: Importância de variáveis (via Random Forest, mais interpretável para isso) ---
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", max_depth=10,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    importancias.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Importância")
    ax.set_title("Top 10 Variáveis Mais Importantes para Prever Abandono")
    fig.tight_layout()
    fig.savefig(DASHBOARDS_DIR / "importancia_variaveis.png", dpi=150)
    plt.close(fig)
    print("Salvo: importancia_variaveis.png")

    # --- Gráfico 3: Trade-off Precision vs Recall por threshold ---
    from sklearn.metrics import precision_score, recall_score

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    precisions, recalls = [], []
    for t in thresholds:
        pred_t = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, pred_t, zero_division=0))
        recalls.append(recall_score(y_test, pred_t))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, marker="o", label="Precision")
    ax.plot(thresholds, recalls, marker="o", label="Recall")
    ax.axvline(x=threshold, color="gray", linestyle="--", label=f"Threshold escolhido ({threshold})")
    ax.set_xlabel("Threshold de decisão")
    ax.set_ylabel("Score")
    ax.set_title("Trade-off: Precision vs Recall por Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DASHBOARDS_DIR / "tradeoff_threshold.png", dpi=150)
    plt.close(fig)
    print("Salvo: tradeoff_threshold.png")


if __name__ == "__main__":
    main()