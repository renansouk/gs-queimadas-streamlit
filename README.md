# Painel de Gestao de Risco de Queimadas

Este diretorio contem uma aplicacao Streamlit para apresentar previsoes de risco de queimadas usando estrategia de lag.

## Arquivos

- `prever_proximas_queimadas.py`: treina modelos com variaveis de periodo anterior e gera previsoes para o proximo mes.
- `app_streamlit.py`: painel interativo para mapa, ranking, filtros e metricas.
- `previsoes_proximo_periodo.csv`: gerado ao executar o script de previsao.

## Como executar o script de previsao

```bash
python prever_proximas_queimadas.py
```

## Como abrir o painel

```bash
streamlit run app_streamlit.py
```

## Logica do modelo

A base e organizada por estado, ano e mes. Para cada estado, o modelo usa informacoes do periodo atual e de periodos anteriores, como:

- focos do mes anterior;
- focos de dois meses anteriores;
- focos do mesmo mes no ano anterior;
- media movel de tres meses;
- media historica daquele estado naquele mes;
- tendencia de crescimento ou queda dos focos;
- variacao em relacao ao mesmo mes do ano anterior;
- temperatura, precipitacao e umidade do periodo de referencia;
- temperatura, precipitacao, umidade e risco do mesmo mes no ano anterior;
- estado, bioma, latitude e longitude.

Com isso, o modelo estima a classe de risco e a quantidade aproximada de focos para o mes seguinte.
