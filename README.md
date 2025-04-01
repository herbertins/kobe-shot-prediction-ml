# 🏀 Kobe Shot Prediction

Este projeto usa técnicas de Machine Learning e engenharia de dados para prever se um arremesso de Kobe Bryant foi **cesta ou erro**, com base em informações contextuais da jogada.

> Projeto desenvolvido seguindo o **Framework TDSP da Microsoft** com foco em boas práticas de MLOps e reprodutibilidade com Kedro.

---

## 🚀 Tecnologias utilizadas

- [Kedro](https://kedro.readthedocs.io/en/stable/) – Orquestração de pipelines
- [PyCaret](https://pycaret.org/) – Treinamento automatizado de modelos
- [MLFlow](https://mlflow.org/) – Rastreamento de experimentos e deployment
- [Scikit-learn](https://scikit-learn.org/) – Métricas e suporte a modelos
- [Streamlit](https://streamlit.io/) – Dashboard interativo
- [Pandas, Parquet, PyArrow] – Manipulação de dados

---

## 📊 Etapas do projeto

| Etapa                          | Descrição |
|-------------------------------|-----------|
| **Aquisição dos dados**       | Importação dos arquivos `.parquet` com as jogadas |
| **Processamento**             | Remoção de dados nulos, seleção de variáveis |
| **Divisão treino/teste**      | Separação estratificada (80/20) |
| **Treinamento**               | Modelos: regressão logística e árvore de decisão |
| **Avaliação**                 | Métricas: Log Loss e F1-Score (registradas no MLFlow) |
| **Deploy**                    | Salvamento e aplicação do modelo final |
| **Dashboard**                 | Interface com Streamlit para análise em produção |

---

## 📁 Estrutura do projeto

```
kobe-shot-prediction-ml/
├── data/                          # Dados (adicionados no .gitignore)
│   ├── 01_raw/                    # Dados brutos (.parquet)
│   ├── 07_model_output/           # Previsões do modelo
├── notebooks/                     # Notebooks exploratórios
├── src/
│   └── kobe_shot_prediction_ml/
│       ├── pipelines/             # Pipelines modulares do Kedro
│       ├── nodes/                 # Funções reutilizáveis
│       └── pipeline_registry.py   # Registro dos pipelines
├── dashboard.py                   # Dashboard Streamlit
├── README.md                      # Este arquivo!
├── requirements.txt               # Dependências
├── pyproject.toml
├── .gitignore
└── mlruns/                        # Experimentos MLFlow (gitignored)
```

---

## 🧪 Resultados

Exemplo de saída da base de produção:

| lat   | lng   | minutes_remaining | prediction_label |
|-------|-------|-------------------|------------------|
| 34.0  | -118  | 5                 | 1 (cesta)        |
| 33.8  | -117  | 2                 | 0 (erro)         |

- F1 Score (produção): `0.71`
- Log Loss (produção): `0.41`

---

## 📊 Dashboard de Monitoramento

Dashboard criado com Streamlit para monitorar os resultados em produção.

### Como executar:

```bash
streamlit run dashboard.py
```

Você poderá visualizar:

- ✅ Taxa de acerto
- 📉 Distribuição das previsões
- 🔍 Amostras das predições feitas
- 🧠 Interface leve e responsiva

---

## 🛠️ Como executar o projeto localmente

1. Clone o repositório
2. Crie um ambiente virtual com Python 3.10+
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute o pipeline completo:

```bash
kedro run
```

5. Inicie o dashboard:

```bash
streamlit run dashboard.py
```

---

## 📚 Referências

- Framework TDSP (Microsoft)
- PyCaret 3.x
- Kedro docs
- MLFlow docs
- Dataset Kobe Bryant [original no Kaggle](https://www.kaggle.com/datasets/kobe24/kobe-bryant-shot-selection)

---

## 💼 Sobre este projeto

Este projeto foi desenvolvido como estudo prático para consolidar conhecimentos em:

- MLOps
- Engenharia de Machine Learning com Kedro
- Rastreabilidade com MLFlow
- Visualização com Streamlit

---

## ✨ Inspiração

> *"The most important thing is to try and inspire people so that they can be great in whatever they want to do."*  
> — Kobe Bryant
