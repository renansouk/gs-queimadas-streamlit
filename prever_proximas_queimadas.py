from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATASET = BASE_DIR / "queimadas_brasil.csv"
PROJECT_DATASET = BASE_DIR.parent / "entrega_tecnica" / "queimadas_brasil.csv"
DEFAULT_DATASET = LOCAL_DATASET if LOCAL_DATASET.exists() else PROJECT_DATASET


FEATURES = [
    "ano",
    "mes",
    "estado_enc",
    "bioma_enc",
    "latitude",
    "longitude",
    "temperatura_media",
    "precipitacao_mm",
    "umidade_relativa",
    "focos_queimadas",
    "focos_lag_1",
    "focos_lag_2",
    "focos_lag_12",
    "risco_lag_1_enc",
    "risco_lag_12_enc",
    "temperatura_lag_1",
    "precipitacao_lag_1",
    "umidade_lag_1",
    "temperatura_lag_12",
    "precipitacao_lag_12",
    "umidade_lag_12",
    "media_focos_3m",
    "media_focos_mes_estado",
    "tendencia_focos",
    "variacao_anual_focos",
]


@dataclass
class LagModelResult:
    classifier: RandomForestClassifier
    regressor: RandomForestRegressor
    encoders: dict[str, LabelEncoder]
    features: list[str]
    metrics: dict[str, float]
    model_data: pd.DataFrame


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["estado", "ano", "mes"]).reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.sort_values(["estado", "ano", "mes"]).copy()

    group = data.groupby("estado", group_keys=False)
    data["focos_lag_1"] = group["focos_queimadas"].shift(1)
    data["focos_lag_2"] = group["focos_queimadas"].shift(2)
    data["focos_lag_12"] = group["focos_queimadas"].shift(12)
    data["temperatura_lag_1"] = group["temperatura_media"].shift(1)
    data["precipitacao_lag_1"] = group["precipitacao_mm"].shift(1)
    data["umidade_lag_1"] = group["umidade_relativa"].shift(1)
    data["temperatura_lag_12"] = group["temperatura_media"].shift(12)
    data["precipitacao_lag_12"] = group["precipitacao_mm"].shift(12)
    data["umidade_lag_12"] = group["umidade_relativa"].shift(12)
    data["risco_lag_1"] = group["risco_queimadas"].shift(1)
    data["risco_lag_12"] = group["risco_queimadas"].shift(12)
    data["media_focos_3m"] = group["focos_queimadas"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    data["tendencia_focos"] = data["focos_lag_1"] - data["focos_lag_2"]
    data["variacao_anual_focos"] = data["focos_queimadas"] - data["focos_lag_12"]

    data["media_focos_mes_estado"] = (
        data.groupby(["estado", "mes"])["focos_queimadas"]
        .transform(lambda values: values.shift(1).expanding().mean())
    )

    data["ano_alvo"] = group["ano"].shift(-1)
    data["mes_alvo"] = group["mes"].shift(-1)
    data["risco_alvo"] = group["risco_queimadas"].shift(-1)
    data["focos_alvo"] = group["focos_queimadas"].shift(-1)

    return data


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    data = add_lag_features(df)

    encoders = {
        "estado": LabelEncoder(),
        "bioma": LabelEncoder(),
        "risco": LabelEncoder(),
        "risco_lag": LabelEncoder(),
        "risco_lag_12": LabelEncoder(),
    }

    data["estado_enc"] = encoders["estado"].fit_transform(data["estado"])
    data["bioma_enc"] = encoders["bioma"].fit_transform(data["bioma"])
    data["risco_alvo_enc"] = encoders["risco"].fit_transform(data["risco_alvo"].fillna("Sem alvo"))
    data["risco_lag_1_enc"] = encoders["risco_lag"].fit_transform(data["risco_lag_1"].fillna("Sem historico"))
    data["risco_lag_12_enc"] = encoders["risco_lag_12"].fit_transform(data["risco_lag_12"].fillna("Sem historico"))

    model_data = data.dropna(subset=FEATURES + ["risco_alvo", "focos_alvo"]).copy()
    model_data["risco_alvo_enc"] = encoders["risco"].transform(model_data["risco_alvo"])
    return model_data, encoders


def train_lag_models(df: pd.DataFrame) -> LagModelResult:
    model_data, encoders = prepare_model_data(df)
    X = model_data[FEATURES]
    y_class = model_data["risco_alvo_enc"]
    y_reg = model_data["focos_alvo"]

    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(X_train, y_train_class)

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )
    regressor.fit(X_train, y_train_reg)

    pred_class = classifier.predict(X_test)
    pred_reg = regressor.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test_class, pred_class),
        "f1_weighted": f1_score(y_test_class, pred_class, average="weighted"),
        "mae_focos": mean_absolute_error(y_test_reg, pred_reg),
        "r2_focos": r2_score(y_test_reg, pred_reg),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
    }

    return LagModelResult(
        classifier=classifier,
        regressor=regressor,
        encoders=encoders,
        features=FEATURES,
        metrics=metrics,
        model_data=model_data,
    )


def next_period(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def build_prediction_base(df: pd.DataFrame, encoders: dict[str, LabelEncoder], year: int, month: int) -> pd.DataFrame:
    lagged = add_lag_features(df)
    base = lagged[(lagged["ano"] == year) & (lagged["mes"] == month)].copy()
    if base.empty:
        raise ValueError(f"Nao existem registros para o periodo {month:02d}/{year}.")

    base["estado_enc"] = encoders["estado"].transform(base["estado"])
    base["bioma_enc"] = encoders["bioma"].transform(base["bioma"])
    base["risco_lag_1_enc"] = encoders["risco_lag"].transform(base["risco_lag_1"].fillna("Sem historico"))
    base["risco_lag_12_enc"] = encoders["risco_lag_12"].transform(base["risco_lag_12"].fillna("Sem historico"))

    missing = base[FEATURES].isna().any(axis=1)
    base = base.loc[~missing].copy()
    if base.empty:
        raise ValueError("O periodo escolhido nao possui historico anterior suficiente para gerar lags.")

    return base


def predict_next_risk(df: pd.DataFrame, result: LagModelResult, year: int, month: int) -> pd.DataFrame:
    base = build_prediction_base(df, result.encoders, year, month)
    target_year, target_month = next_period(year, month)

    X_next = base[result.features]
    pred_class = result.classifier.predict(X_next)
    pred_proba = result.classifier.predict_proba(X_next)
    pred_focos = result.regressor.predict(X_next)

    class_names = result.encoders["risco"].inverse_transform(pred_class)
    proba_max = pred_proba.max(axis=1)

    output = base[
        [
            "estado",
            "bioma",
            "latitude",
            "longitude",
            "ano",
            "mes",
            "focos_queimadas",
            "risco_queimadas",
            "focos_lag_1",
            "focos_lag_12",
            "media_focos_3m",
            "media_focos_mes_estado",
            "tendencia_focos",
            "variacao_anual_focos",
            "temperatura_media",
            "precipitacao_mm",
            "umidade_relativa",
        ]
    ].copy()
    output = output.rename(
        columns={
            "ano": "ano_referencia",
            "mes": "mes_referencia",
            "focos_queimadas": "focos_periodo_referencia",
            "risco_queimadas": "risco_periodo_referencia",
        }
    )
    output["ano_previsto"] = target_year
    output["mes_previsto"] = target_month
    output["risco_previsto"] = class_names
    output["confianca"] = proba_max
    output["focos_previstos"] = np.maximum(pred_focos, 0).round(0).astype(int)

    risk_order = {"Alto": 3, "Médio": 2, "Medio": 2, "Baixo": 1}
    output["prioridade"] = output["risco_previsto"].map(risk_order).fillna(0)
    output = output.sort_values(["prioridade", "focos_previstos", "confianca"], ascending=False)
    return output.reset_index(drop=True)


def generate_default_predictions(dataset_path: str | Path = DEFAULT_DATASET) -> tuple[pd.DataFrame, LagModelResult]:
    df = load_dataset(dataset_path)
    result = train_lag_models(df)
    last = df.sort_values(["ano", "mes"]).iloc[-1]
    predictions = predict_next_risk(df, result, int(last["ano"]), int(last["mes"]))
    return predictions, result


def main() -> None:
    predictions, result = generate_default_predictions()
    out = BASE_DIR / "previsoes_proximo_periodo.csv"
    predictions.to_csv(out, index=False, encoding="utf-8")

    print("Modelo de lag treinado com sucesso.")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"F1-score ponderado: {result.metrics['f1_weighted']:.4f}")
    print(f"MAE focos previstos: {result.metrics['mae_focos']:.2f}")
    print(f"Arquivo gerado: {out}")
    print()
    print("Top 10 locais prioritarios:")
    print(predictions[["estado", "bioma", "ano_previsto", "mes_previsto", "risco_previsto", "focos_previstos", "confianca"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
