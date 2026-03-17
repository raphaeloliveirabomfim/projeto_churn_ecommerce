# Dados — Churn E-commerce

## Arquivos

### `data/raw/` — Dados brutos
| Arquivo | Descrição | Registros |
|---------|-----------|-----------|
| `clientes.csv` | Perfil demográfico, canal de aquisição e label de churn | 10.000 |
| `transacoes.csv` | Histórico completo de compras (2020–2024) | ~72.000 |

### `data/processed/` — Gerados pelos notebooks
| Arquivo | Gerado por |
|---------|-----------|
| `X_features.csv` | `02_feature_engineering.ipynb` |
| `y_target.csv` | `02_feature_engineering.ipynb` |
| `clientes_com_features.csv` | `02_feature_engineering.ipynb` |
| `clientes_segmentados.csv` | `03_modelagem.ipynb` |

---

## Dicionário — `clientes.csv`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id_cliente` | int | Identificador único |
| `idade` | int | Idade em anos |
| `genero` | str | M ou F |
| `cidade` | str | SP, RJ, BH, BSB, POA |
| `data_cadastro` | date | Data de registro na plataforma |
| `canal_aquisicao` | str | Google, Facebook, Indicação, Orgânico |
| `churn` | int | **Target** — 0: ativo, 1: cancelou |

## Dicionário — `transacoes.csv`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id_cliente` | int | Referência ao cliente |
| `data_compra` | datetime | Data e hora da compra |
| `valor` | float | Valor em R$ |
| `categoria` | str | Eletrônicos, Moda, Casa, Esportes, Beleza |
| `metodo_pagamento` | str | Cartão, Pix, Boleto |
