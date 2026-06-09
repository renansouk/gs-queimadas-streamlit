from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from prever_proximas_queimadas import DEFAULT_DATASET, load_dataset, predict_next_risk, train_lag_models


st.set_page_config(
    page_title="ÍGNIS - Risco de Queimadas",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        visibility: hidden;
        height: 0;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 247, 243, 0.96);
    }
    .block-container {
        padding-top: 2.25rem;
    }
    .tool-name {
        font-size: 0.95rem;
        letter-spacing: 0.16rem;
        text-transform: uppercase;
        color: #5f7f6a;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .tool-title {
        font-size: clamp(2.1rem, 4vw, 3.2rem);
        line-height: 1.05;
        font-weight: 800;
        color: #202020;
        margin-bottom: 0.6rem;
    }
    .tool-subtitle {
        color: #6f756f;
        max-width: 860px;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


RISK_COLORS = {
    "Baixo": "#4a7c59",
    "Médio": "#d9a441",
    "Medio": "#d9a441",
    "Alto": "#b7352d",
}


@st.cache_data
def cached_dataset(path: str, mtime: float) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_resource
def cached_model(path: str, mtime: float):
    df = load_dataset(path)
    return train_lag_models(df)


def month_label(month: int) -> str:
    labels = {
        1: "Janeiro",
        2: "Fevereiro",
        3: "Março",
        4: "Abril",
        5: "Maio",
        6: "Junho",
        7: "Julho",
        8: "Agosto",
        9: "Setembro",
        10: "Outubro",
        11: "Novembro",
        12: "Dezembro",
    }
    return labels[int(month)]


def render_header():
    st.markdown(
        """
        <div class="tool-name">ÍGNIS</div>
        <div class="tool-title">Painel de Gestão de Risco de Queimadas</div>
        <div class="tool-subtitle">
        Inteligência geoespacial para prever o risco do próximo mês por estado e bioma,
        priorizando ações antes do pico de seca.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame):
    st.sidebar.header("Período de referência")
    years = sorted(df["ano"].unique())
    year = st.sidebar.selectbox("Ano", years, index=len(years) - 1)

    available_months = sorted(df.loc[df["ano"] == year, "mes"].unique())
    month = st.sidebar.selectbox(
        "Mês",
        available_months,
        index=len(available_months) - 1,
        format_func=month_label,
    )

    st.sidebar.header("Filtros")
    biomas = ["Todos"] + sorted(df["bioma"].unique())
    bioma = st.sidebar.selectbox("Bioma", biomas)

    riscos = ["Todos", "Alto", "Médio", "Baixo"]
    risco = st.sidebar.selectbox("Risco previsto", riscos)

    min_conf = st.sidebar.slider("Confiança mínima", 0.0, 1.0, 0.0, 0.05)
    return int(year), int(month), bioma, risco, min_conf


def filter_predictions(predictions: pd.DataFrame, bioma: str, risco: str, min_conf: float) -> pd.DataFrame:
    data = predictions.copy()
    if bioma != "Todos":
        data = data[data["bioma"] == bioma]
    if risco != "Todos":
        data = data[data["risco_previsto"] == risco]
    data = data[data["confianca"] >= min_conf]
    return data


def render_metrics(predictions: pd.DataFrame, metrics: dict[str, float]):
    high_count = int((predictions["risco_previsto"] == "Alto").sum())
    total_focos = int(predictions["focos_previstos"].sum())
    avg_conf = predictions["confianca"].mean() if len(predictions) else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estados em risco alto", high_count)
    col2.metric("Focos previstos", f"{total_focos:,}".replace(",", "."))
    col3.metric("Confiança média", f"{avg_conf:.1%}")
    col4.metric("F1 do modelo", f"{metrics['f1_weighted']:.3f}")


def render_map(predictions: pd.DataFrame):
    if predictions.empty:
        st.warning("Nenhum estado encontrado com os filtros selecionados.")
        return

    fig = px.scatter_map(
        predictions,
        lat="latitude",
        lon="longitude",
        color="risco_previsto",
        size="focos_previstos",
        hover_name="estado",
        hover_data={
            "bioma": True,
            "risco_previsto": True,
            "focos_previstos": True,
            "confianca": ":.1%",
            "latitude": False,
            "longitude": False,
        },
        color_discrete_map=RISK_COLORS,
        zoom=3.1,
        height=520,
    )
    fig.update_layout(
        map_style="open-street-map",
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="Risco previsto",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_charts(predictions: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        ranking = predictions.sort_values("focos_previstos", ascending=True).tail(12)
        fig = px.bar(
            ranking,
            x="focos_previstos",
            y="estado",
            color="risco_previsto",
            orientation="h",
            color_discrete_map=RISK_COLORS,
            title="Ranking de focos previstos",
        )
        fig.update_layout(xaxis_title="Focos previstos", yaxis_title="", legend_title_text="Risco")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_risk = predictions.groupby("risco_previsto", as_index=False).agg(
            estados=("estado", "count"),
            focos_previstos=("focos_previstos", "sum"),
        )
        fig = px.bar(
            by_risk,
            x="risco_previsto",
            y="focos_previstos",
            color="risco_previsto",
            color_discrete_map=RISK_COLORS,
            title="Focos previstos por classe de risco",
        )
        fig.update_layout(xaxis_title="Risco previsto", yaxis_title="Focos previstos", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_table(predictions: pd.DataFrame):
    cols = [
        "estado",
        "bioma",
        "ano_previsto",
        "mes_previsto",
        "risco_previsto",
        "focos_previstos",
        "confianca",
        "focos_periodo_referencia",
        "focos_lag_12",
        "media_focos_3m",
        "media_focos_mes_estado",
        "tendencia_focos",
        "variacao_anual_focos",
        "temperatura_media",
        "precipitacao_mm",
        "umidade_relativa",
    ]
    table = predictions[cols].copy()
    table["confianca"] = table["confianca"].map(lambda value: f"{value:.1%}")
    table["media_focos_3m"] = table["media_focos_3m"].round(1)
    table["media_focos_mes_estado"] = table["media_focos_mes_estado"].round(1)
    table["tendencia_focos"] = table["tendencia_focos"].round(1)
    table["variacao_anual_focos"] = table["variacao_anual_focos"].round(1)
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_explanation(year: int, month: int, target_year: int, target_month: int):
    st.subheader("Como interpretar")
    st.write(
        f"""
        O painel usa o período de referência {month_label(month)} de {year} para estimar o risco de
        {month_label(target_month)} de {target_year}. A estratégia de lag considera informações do mês atual
        e de meses anteriores, como focos recentes, média móvel de três meses, tendência de crescimento,
        temperatura, chuva e umidade.

        Na prática, o modelo tenta responder: considerando o histórico recente de cada estado, quais locais
        têm maior probabilidade de apresentar risco elevado no próximo mês?
        """
    )


def main():
    dataset_path = Path(DEFAULT_DATASET)
    render_header()

    if not dataset_path.exists():
        st.error(f"Dataset não encontrado: {dataset_path}")
        st.stop()

    dataset_mtime = dataset_path.stat().st_mtime
    df = cached_dataset(str(dataset_path), dataset_mtime)
    result = cached_model(str(dataset_path), dataset_mtime)

    year, month, bioma, risco, min_conf = render_sidebar(df)
    predictions = predict_next_risk(df, result, year, month)
    filtered = filter_predictions(predictions, bioma, risco, min_conf)

    if predictions.empty:
        st.error("Não foi possível gerar previsões para o período selecionado.")
        st.stop()

    target_year = int(predictions["ano_previsto"].iloc[0])
    target_month = int(predictions["mes_previsto"].iloc[0])

    st.subheader(f"Previsão para {month_label(target_month)} de {target_year}")
    render_metrics(filtered, result.metrics)

    tab_map, tab_charts, tab_table, tab_model = st.tabs(
        ["Mapa de risco", "Indicadores", "Tabela de priorização", "Modelo"]
    )

    with tab_map:
        render_map(filtered)

    with tab_charts:
        render_charts(filtered)

    with tab_table:
        render_table(filtered)
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar previsões em CSV",
            csv,
            file_name="previsoes_risco_queimadas.csv",
            mime="text/csv",
        )

    with tab_model:
        render_explanation(year, month, target_year, target_month)
        st.subheader("Desempenho")
        metrics_df = pd.DataFrame(
            [
                {"Métrica": "Acurácia da classificação", "Valor": f"{result.metrics['accuracy']:.4f}"},
                {"Métrica": "F1-score ponderado", "Valor": f"{result.metrics['f1_weighted']:.4f}"},
                {"Métrica": "Erro médio absoluto dos focos", "Valor": f"{result.metrics['mae_focos']:.2f}"},
                {"Métrica": "R2 da previsão de focos", "Valor": f"{result.metrics['r2_focos']:.4f}"},
                {"Métrica": "Registros de treino", "Valor": f"{int(result.metrics['train_rows'])}"},
                {"Métrica": "Registros de teste", "Valor": f"{int(result.metrics['test_rows'])}"},
            ]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
