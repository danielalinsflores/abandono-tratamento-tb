from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import joblib

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "tb_rj_features.csv")

    X = df.drop(columns=["abandonou", "NU_ANO"])
    y = df["abandonou"]

    # stratify=y garante que a proporção de abandono (19,4%) seja mantida
    # tanto no treino quanto no teste - crucial com classes desbalanceadas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Treino: {len(X_train)} linhas ({y_train.mean()*100:.1f}% abandono)")
    print(f"Teste: {len(X_test)} linhas ({y_test.mean()*100:.1f}% abandono)")

    # Baseline: Regressão Logística com class_weight="balanced" para
    # compensar o desbalanceamento (19% positivo vs 81% negativo)
    modelo = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print("\n--- Relatório de Classificação (Regressão Logística) ---")
    print(classification_report(y_test, y_pred, target_names=["Não abandonou", "Abandonou"]))

    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    MODELS_DIR.mkdir(exist_ok=True)

    from sklearn.ensemble import RandomForestClassifier

    print("\n\n--- Relatório de Classificação (Random Forest) ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_rf, target_names=["Não abandonou", "Abandonou"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.3f}")
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred_rf))

    # Importância das variáveis - ótimo para interpretação e storytelling
    importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Top 10 variáveis mais importantes ---")
    print(importancias.head(10))

    print("\n\n--- Ajuste de threshold (Regressão Logística) ---")
    for threshold in [0.5, 0.4, 0.35, 0.3]:
        y_pred_ajustado = (y_proba >= threshold).astype(int)
        print(f"\nThreshold = {threshold}")
        print(classification_report(
            y_test, y_pred_ajustado,
            target_names=["Não abandonou", "Abandonou"],
            zero_division=0,
        ))

    from imblearn.over_sampling import SMOTE

    print("\n\n--- Regressão Logística com SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"Treino original: {len(X_train)} linhas ({y_train.mean()*100:.1f}% abandono)")
    print(f"Treino com SMOTE: {len(X_train_smote)} linhas ({y_train_smote.mean()*100:.1f}% abandono)")

    modelo_smote = LogisticRegression(max_iter=1000, random_state=42)
    modelo_smote.fit(X_train_smote, y_train_smote)

    y_pred_smote = modelo_smote.predict(X_test)
    y_proba_smote = modelo_smote.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_smote, target_names=["Não abandonou", "Abandonou"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_smote):.3f}")
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred_smote))

    THRESHOLD_FINAL = 0.4
    joblib.dump(modelo, MODELS_DIR / "modelo_regressao_logistica.pkl")
    with open(MODELS_DIR / "threshold.txt", "w") as f:
        f.write(str(THRESHOLD_FINAL))
    print(f"\nModelo salvo em {MODELS_DIR / 'modelo_regressao_logistica.pkl'}")


if __name__ == "__main__":
    main()