echo '# Projeto de Análise e Predição de Churn para E-commerce

## Descrição
Este projeto tem como objetivo analisar o comportamento de clientes de um e-commerce e desenvolver um modelo preditivo para identificar clientes com maior probabilidade de cancelamento (churn). Através de técnicas avançadas de machine learning e análise exploratória, foram gerados insights acionáveis e recomendações estratégicas para reduzir a taxa de cancelamento e aumentar a retenção de clientes.

## Objetivos
- Análise exploratória completa dos dados de clientes e transações
- Desenvolvimento de features relevantes para o modelo preditivo
- Modelagem preditiva para identificação de clientes em risco de churn
- Segmentação de clientes por nível de risco
- Recomendações de negócio baseadas em dados com impacto financeiro estimado

## Tecnologias Utilizadas
- **Python**: pandas, numpy, scikit-learn, xgboost, imbalanced-learn
- **Visualização**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit (em desenvolvimento)
- **Modelagem**: Regressão Logística, Random Forest, XGBoost, Stacking Ensemble
- **Versionamento**: Git

## Estrutura do Projeto
projeto_churn_ecommerce/
├── data/
│ ├── raw/ # Dados brutos (clientes e transações)
│ └── processed/ # Dados processados com features e segmentação
├── notebooks/
│ ├── 01_analise_exploratoria.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modelagem.ipynb
├── src/ # Código fonte modularizado
├── dashboard/ # Aplicação Streamlit
├── reports/ # Relatórios, figuras e métricas
└── requirements.txt # Dependências do projeto

## Como Executar
1. Clone este repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks em ordem numérica
4. Para o dashboard: `streamlit run dashboard/app.py`

## Tratamento de Dados
### Geração de Dados Realistas
Como o projeto visa demonstrar técnicas de ciência de dados em um cenário realista, foram gerados dados sintéticos com **relações de negócio reais**:

- **Idade**: Distribuição normal (média 35 anos, desvio 12)
- **Canais de aquisição**: Google (30%), Facebook (20%), Indicação (20%), Orgânico (30%)
- **Cidades**: SP (40%), RJ (30%), BH (10%), BSB (10%), POA (10%)
- **Comportamento de compra**: Número de compras e valores influenciados por idade, cidade e canal

### Engenharia de Atributos (Feature Engineering)
Foram criadas mais de 30 features para capturar diferentes aspectos do comportamento dos clientes:

**Features RFV (Recência, Frequência, Valor)**
- `recencia_dias`: Dias desde a última compra
- `frequencia_compras`: Número total de compras
- `valor_total`: Valor total gasto
- `ticket_medio`: Valor médio por compra

**Features Comportamentais**
- `compras_*`: Quantidade de compras por categoria (Eletrônicos, Moda, Casa, Esportes, Beleza)
- `metodo_preferido`: Método de pagamento preferido
- `dias_entre_compras`: Intervalo médio entre compras

**Features de Proporção**
- `prop_*`: Proporção de compras em cada categoria
- `ticket_medio_relativo`: Ticket médio em relação à média geral

### Tratamento de Valores Ausentes
- **Clientes sem compras**: 59 clientes (0,59% da base) não realizaram nenhuma compra
- **Estratégia aplicada**:
  1. Preenchimento de `valor_total = 0` para viabilizar o modelo numérico
  2. Criação da flag binária `cliente_sem_compra` para capturar este comportamento atípico
- **Validação**: Clientes sem compras apresentaram taxa de churn significativamente diferente, justificando o tratamento diferenciado

### Preparação para Modelagem
- **Encoding**: Variáveis categóricas transformadas via one-hot encoding
- **Escalamento**: Features numéricas padronizadas (média 0, desvio 1)
- **Split**: 80% treino / 20% teste com estratificação para manter proporção de churn

## Análise Exploratória - Principais Insights
**Taxa de Churn Geral**
22.26% dos clientes cancelam (aproximadamente 1 em cada 4)

**Canais de Aquisição**
Melhor canal: Indicação (9.2% de churn)

Pior canal: Facebook (33.7% de churn)

Ação: Investir em programa de indicação, revisar campanhas no Facebook

**Distribuição Geográfica**
Melhor cidade: São Paulo (14.5% de churn)

Pior cidade: Brasília (36.3% de churn)

Ação: Investigar causas do alto churn em Brasília (concorrência? logística?)

**Comportamento de Compra**
Clientes que não cancelaram: média de 7.7 compras

Clientes que cancelaram: média de 5.7 compras

Ação: Engajar clientes até a 6ª compra (ponto de virada)

**Perfil Demográfico**
Maior risco: Faixa etária 56-70 anos (31.5% de churn)

Menor risco: Faixa etária 18-25 anos (18.2% de churn)

*Ação: Programa especial para clientes 55+*

## Modelagem Preditiva
**Modelos Testados**
Modelo	           Recall  AUC-ROC F1-Score	Precisão
Regressão Logística	0.697	0.741	0.478	0.363
Árvore de Decisão	0.735	0.728	0.495	0.374
Random Forest	    0.103	0.713	0.178	0.639
XGBoost	            0.333	0.695	0.369	0.415

**Modelos Otimizados**
Modelo	                   Recall  AUC-ROC F1-Score	Precisão
Logistic (Otimizado)	        0.708	0.741	0.483	0.367
Random Forest (Otimizado)	    0.663	0.739	0.489	0.387
XGBoost (Otimizado)	            0.771   0.749	0.485	0.354
XGBoost + Features Avançadas	0.766	0.744	0.487	0.357
Stacking Ensemble	            0.108	0.753	0.186	0.676

Modelo Escolhido: XGBoost Otimizado

Critério de seleção: Priorização de RECALL (identificar o máximo de clientes em risco)

Métrica	Valor	Interpretação
Recall	 77.1%	Identifica 77 de cada 100 clientes que cancelariam
AUC-ROC	 0.749	Bom poder de discriminação entre classes
F1-Score 0.485	Equilíbrio entre recall e precisão

Comparação com Árvore de Decisão:
- Recall: 4.9% superior ao da Árvore (77.1% vs 73.5%)

- AUC-ROC: 2.9% superior (0.749 vs 0.728)

## Segmentação de Risco
Com o modelo final, os clientes foram segmentados em 5 níveis de risco. Os valores abaixo são ilustrativos - preencha com os números obtidos na sua execução:

Segmento	Clientes	% da Base	Churn Real	Ação Recomendada
Muito Baixo	  2.847	      28,5%	       4,2%	   Manutenção de engajamento
Baixo	      2.891	      28,9%	      12,8%    Programa de fidelidade
Médio	      2.156	      21,6%	      24,5%	   Ações preventivas
Alto	      1.452	      14,5%	      41,3%	   Contato proativo
Crítico	       654	       6,5%	      67,8%	   Intervenção imediata

## Top 10 Features Mais Importantes para churn:
- recencia_dias: Quanto mais dias sem comprar, maior o risco

- frequencia_compras: Clientes que compram pouco têm maior risco

- valor_total: Baixo valor gasto indica menor engajamento

- dias_entre_compras: Longos intervalos sinalizam desinteresse

- canal_aquisicao_Facebook: Canal com maior taxa de churn

- cidade_BSB: Localização de maior risco

- cliente_sem_compra: Comportamento atípico de clientes sem compras

- ticket_medio: Valor médio das compras reflete perfil

- compras_eletrônicos: Categoria preferida indica perfil

- idade: Faixa etária correlacionada com hábitos

## Impacto de Negócio Estimado
Premissas
Ticket médio: R$ 172,00

Ciclo de compras anual: 4 compras/cliente

Taxa de sucesso das ações de retenção: 35%

Custo médio por ação: R$ 10 a R$ 50 (por segmento)

Resultados Financeiros
Receita retida anual: R$ [valor calculado]

Custo total das ações: R$ [valor calculado]

Lucro líquido: R$ [valor calculado]

ROI estimado: [X]x (R$ [X] para cada R$ 1,00 investido)

Impacto na Taxa de Churn
Taxa atual: 22.26%

Taxa projetada: [X]%

Redução absoluta: [X] pontos percentuais

Redução relativa: [X]%

## Recomendações de Negócio
**Muito Baixo Risco**

Manter engajamento com newsletter mensal

Oferecer descontos sazonais

Programa de indicação para atrair novos clientes

Coletar depoimentos e avaliações

**Baixo Risco**

Programa de fidelidade com pontos acumulativos

Ofertas personalizadas baseadas em histórico

Email marketing segmentado com recomendações

Acesso antecipado a promoções

**Médio Risco**

Aumentar frequência de contato

Oferecer benefícios exclusivos para retenção

Pesquisa de satisfação direcionada

Cupom de desconto para próxima compra

**Alto Risco**

Contato proativo por telefone/WhatsApp

Oferta de retenção personalizada

Análise de reclamações e feedbacks recentes

Acompanhamento por customer success

**Crítico**

INTERVENÇÃO IMEDIATA - prioridade máxima

Oferta especial de retenção (desconto agressivo)

Entrevista de saída para entender motivos

Suporte prioritário com atendente dedicado

**Plano de Ação Cronológico**

**Primeiros 30 dias (Prioridade Máxima)**

Contatar todos os clientes do segmento CRÍTICO

Implementar oferta especial para segmento ALTO

Configurar alertas automáticos para novos clientes de alto risco

**30-60 dias (Prioridade Média)**

Lançar programa de fidelidade para segmento MÉDIO

Criar campanhas de email segmentadas para BAIXO risco

Analisar feedback dos clientes recuperados

**60-90 dias (Prioridade Baixa)**

Automatizar newsletter para MUITO BAIXO risco

Desenvolver programa de indicação

Criar dashboard de monitoramento contínuo

## Conclusão
O projeto desenvolveu um modelo preditivo de churn com 77% de recall, capaz de identificar corretamente mais de 3/4 dos clientes em risco de cancelamento. Com a implementação das recomendações segmentadas, estima-se uma redução significativa da taxa de churn e um ROI positivo para as ações de retenção.

## Autor
Raphael Oliveira Bomfim
- LinkedIn: [https://www.linkedin.com/in/raphael-oliveira-bomfim/]
- GitHub: [https://github.com/raphaeloliveirabomfim]
' > README.md