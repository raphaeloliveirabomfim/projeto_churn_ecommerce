# Previsão de Churn em E-commerce

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-blue)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Visão Geral

Modelo preditivo para identificar clientes com alta probabilidade de cancelamento (churn) em um e-commerce, permitindo ações de retenção proativas e personalizadas por segmento de risco — antes que o cancelamento aconteça.

> Taxa de churn atual: **~22%** — aproximadamente 1 em cada 4 clientes cancela.

---

## Problema de Negócio

A abordagem reativa tradicional (cliente cancela → empresa tenta reativar) tem baixo ROI. Com ML, mudamos para uma abordagem **preditiva e proativa**:

> *"Quais clientes têm maior probabilidade de cancelar nos próximos 30-60 dias?"*

**Prioridade:** Maximizar Recall — cada cliente de churn não identificado é receita perdida.

---

## Estrutura do Projeto

```
churn-ecommerce/
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Análise exploratória completa
│   ├── 02_feature_engineering.ipynb   # Construção de features RFV + comportamentais
│   └── 03_modelagem.ipynb             # Pipeline, modelos, segmentação e ROI
│
├── data/
│   ├── raw/                           # clientes.csv, transacoes.csv
│   ├── processed/                     # Features e segmentação geradas pelos notebooks
│   └── README.md
│
├── figures/                         # Gráficos gerados automaticamente
│
├── models/
│   ├── pipeline_final.pkl             # Pipeline completo (scaler + SMOTE + modelo)
│   ├── threshold_final.pkl            # Threshold otimizado por F2-Score
│   ├── feature_importance.csv         # Importância das features
│   └── resultados_modelos.csv         # Tabela comparativa
│
├── dashboard/
│   └── app.py                         # Dashboard Streamlit
│
├── requirements.txt
└── README.md
```
> *O arquivo 'pipeline_final' ficou acima dos 25MB e por isso não está no repositório

---

## Dados

Dataset sintético baseado em padrões reais de e-commerce brasileiro.

| Arquivo | Descrição | Registros |
|---------|-----------|-----------|
| `clientes.csv` | Perfil demográfico e canal de aquisição | 10.000 |
| `transacoes.csv` | Histórico de compras (2020–2024) | ~72.000 |

---

## Metodologia

### Features criadas (30+)

| Grupo | Features |
|-------|----------|
| **RFV** | `recencia_dias`, `frequencia_compras`, `valor_total`, `ticket_medio` |
| **Comportamental** | `compras_*` por categoria, `metodo_preferido`, `dias_entre_compras` |
| **Proporção** | `prop_compras_*` — mix de categorias do cliente |
| **Engenharia** | `ticket_medio_rel`, `cliente_sem_compra`, `std_valor` |

### Diferenciais técnicos

- **SMOTE dentro do pipeline** — evita data leakage (problema do notebook original)
- **Threshold otimizado por F2-Score** — resolve o RF com 10% de Recall do projeto original
- **Precision-Recall Curve** como métrica principal (mais adequada para dados desbalanceados)
- **Segmentação em 5 níveis** com ações recomendadas por segmento

---

## Resultados

| Modelo | Threshold | Recall | F2-Score | AUC-ROC |
|--------|-----------|--------|----------|---------|
| Logistic Regression | 0.344 | 0.896 | 0.640 | 0.737 |
| Random Forest | 0.262 | 0.867 | 0.644 | 0.725 |
| XGBoost | 0.372 | 0.836 | 0637 | 0.715 |
| LightGBM | 0.272 | 0.862 | 0.631 | 0.713 |

---

## Segmentação de Risco

| Segmento | % da Base | Churn Real | Ação |
|----------|-----------|------------|------|
| Muito Baixo | ~28% | ~4% | Newsletter + programa de indicação |
| Baixo | ~29% | ~13% | Programa de fidelidade |
| Médio | ~22% | ~25% | Cupom + pesquisa de satisfação |
| Alto | ~15% | ~41% | Contato proativo por WhatsApp |
| Crítico | ~6% | ~68% | Intervenção imediata + oferta agressiva |

---

## Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/churn-ecommerce.git
cd churn-ecommerce

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute os notebooks em ordem
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modelagem.ipynb

# 4. Rode o dashboard
streamlit run dashboard/app.py
```

---

## Autor

**Raphael Oliveira Bomfim**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/raphael-oliveira-bomfim-73b808260/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://https://github.com/raphaeloliveirabomfim/)

