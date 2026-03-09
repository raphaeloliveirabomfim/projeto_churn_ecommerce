# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Churn - E-commerce",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🛍️ Dashboard de Análise e Predição de Churn")
st.markdown("---")

# ============================================
# CARREGAMENTO DOS DADOS
# ============================================

@st.cache_data
def carregar_dados():
    """Carrega todos os dados necessários para o dashboard"""
    try:
        # Dados processados
        clientes = pd.read_csv('data/processed/clientes_com_features.csv')
        transacoes = pd.read_csv('data/raw/transacoes.csv')
        
        # Converter datas
        if 'data_compra' in transacoes.columns:
            transacoes['data_compra'] = pd.to_datetime(transacoes['data_compra'])
        
        # Verificar se temos o arquivo com segmentação
        if os.path.exists('data/processed/clientes_segmentados_final.csv'):
            clientes_segmentados = pd.read_csv('data/processed/clientes_segmentados_final.csv')
            # Mesclar com clientes originais se necessário
            for col in clientes_segmentados.columns:
                if col not in clientes.columns:
                    clientes[col] = clientes_segmentados[col]
        
        return clientes, transacoes
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

@st.cache_data
def carregar_metricas():
    """Carrega as métricas dos modelos"""
    try:
        if os.path.exists('reports/metricas_finais.csv'):
            metricas = pd.read_csv('reports/metricas_finais.csv', index_col=0)
            return metricas
        else:
            # Dados mockados para demonstração
            metricas = pd.DataFrame({
                'Modelo': ['Regressão Logística', 'Árvore de Decisão', 'Random Forest', 'XGBoost', 'XGBoost Otimizado'],
                'Recall': [0.697, 0.735, 0.103, 0.333, 0.771],
                'AUC-ROC': [0.741, 0.728, 0.713, 0.695, 0.749],
                'F1-Score': [0.478, 0.495, 0.178, 0.369, 0.485],
                'Precisão': [0.363, 0.374, 0.639, 0.415, 0.354]
            }).set_index('Modelo')
            return metricas
    except:
        return None

@st.cache_data
def carregar_importancias():
    """Carrega a importância das features"""
    try:
        if os.path.exists('reports/feature_importance_final.csv'):
            importancias = pd.read_csv('reports/feature_importance_final.csv')
            return importancias
        else:
            return None
    except:
        return None

# Carregar dados
clientes, transacoes = carregar_dados()
metricas = carregar_metricas()
importancias = carregar_importancias()

# ============================================
# SIDEBAR - FILTROS
# ============================================

st.sidebar.header("🔍 Filtros")

if clientes is not None:
    # Filtro por cidade
    cidades = ['Todas'] + sorted(clientes['cidade'].unique().tolist())
    cidade_selecionada = st.sidebar.selectbox("Cidade", cidades)
    
    # Filtro por canal de aquisição
    canais = ['Todos'] + sorted(clientes['canal_aquisicao'].unique().tolist())
    canal_selecionado = st.sidebar.selectbox("Canal de Aquisição", canais)
    
    # Filtro por faixa etária
    idade_min = int(clientes['idade'].min())
    idade_max = int(clientes['idade'].max())
    idade_range = st.sidebar.slider("Faixa Etária", idade_min, idade_max, (idade_min, idade_max))
    
    # Filtro por segmento de risco
    if 'segmento_detalhado' in clientes.columns:
        segmentos = ['Todos'] + sorted(clientes['segmento_detalhado'].unique().tolist())
        segmento_selecionado = st.sidebar.selectbox("Segmento de Risco", segmentos)
    else:
        segmento_selecionado = 'Todos'
    
    # Aplicar filtros
    dados_filtrados = clientes.copy()
    
    if cidade_selecionada != 'Todas':
        dados_filtrados = dados_filtrados[dados_filtrados['cidade'] == cidade_selecionada]
    
    if canal_selecionado != 'Todos':
        dados_filtrados = dados_filtrados[dados_filtrados['canal_aquisicao'] == canal_selecionado]
    
    dados_filtrados = dados_filtrados[
        (dados_filtrados['idade'] >= idade_range[0]) & 
        (dados_filtrados['idade'] <= idade_range[1])
    ]
    
    if segmento_selecionado != 'Todos' and 'segmento_detalhado' in dados_filtrados.columns:
        dados_filtrados = dados_filtrados[dados_filtrados['segmento_detalhado'] == segmento_selecionado]
    
    # Métricas rápidas na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Métricas Rápidas")
    st.sidebar.metric("Total Clientes", f"{len(dados_filtrados):,}")
    st.sidebar.metric("Taxa de Churn", f"{dados_filtrados['churn'].mean()*100:.2f}%")
else:
    st.sidebar.warning("Dados não carregados")

# ============================================
# PRINCIPAIS MÉTRICAS (KPIs)
# ============================================

if clientes is not None:
    st.subheader("📈 Principais Indicadores")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Taxa de Churn Geral",
            f"{clientes['churn'].mean()*100:.2f}%",
            delta=f"{clientes['churn'].mean()*100 - 20:.2f}pp"  # Comparação com benchmark de 20%
        )
    
    with col2:
        melhor_canal = clientes.groupby('canal_aquisicao')['churn'].mean().idxmin()
        melhor_taxa = clientes.groupby('canal_aquisicao')['churn'].mean().min() * 100
        st.metric(
            "Melhor Canal",
            melhor_canal,
            delta=f"{melhor_taxa:.2f}% churn"
        )
    
    with col3:
        pior_canal = clientes.groupby('canal_aquisicao')['churn'].mean().idxmax()
        pior_taxa = clientes.groupby('canal_aquisicao')['churn'].mean().max() * 100
        st.metric(
            "Pior Canal",
            pior_canal,
            delta=f"{pior_taxa:.2f}% churn",
            delta_color="inverse"
        )
    
    with col4:
        if 'probabilidade_final' in clientes.columns:
            risco_medio = clientes['probabilidade_final'].mean() * 100
            st.metric(
                "Risco Médio",
                f"{risco_medio:.2f}%",
                delta=f"{risco_medio - clientes['churn'].mean()*100:.2f}pp vs real"
            )
        else:
            ticket_medio = clientes['ticket_medio'].mean()
            st.metric(
                "Ticket Médio",
                f"R$ {ticket_medio:.2f}"
            )
    
    st.markdown("---")

# ============================================
# ABAS DO DASHBOARD
# ============================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Análise Exploratória", 
    "🤖 Modelos Preditivos", 
    "📈 Segmentação de Risco",
    "🔑 Importância de Features",
    "💡 Recomendações"
])

# ============================================
# TAB 1 - ANÁLISE EXPLORATÓRIA
# ============================================

with tab1:
    st.subheader("📊 Análise Exploratória dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de churn por canal
        st.markdown("**Taxa de Churn por Canal de Aquisição**")
        churn_canal = clientes.groupby('canal_aquisicao')['churn'].mean().sort_values() * 100
        fig_canal = px.bar(
            x=churn_canal.values,
            y=churn_canal.index,
            orientation='h',
            color=churn_canal.values,
            color_continuous_scale=['green', 'yellow', 'red'],
            labels={'x': 'Taxa de Churn (%)', 'y': 'Canal'}
        )
        fig_canal.update_layout(height=350)
        st.plotly_chart(fig_canal, use_container_width=True)
    
    with col2:
        # Gráfico de churn por cidade
        st.markdown("**Taxa de Churn por Cidade**")
        churn_cidade = clientes.groupby('cidade')['churn'].mean().sort_values() * 100
        fig_cidade = px.bar(
            x=churn_cidade.values,
            y=churn_cidade.index,
            orientation='h',
            color=churn_cidade.values,
            color_continuous_scale=['green', 'yellow', 'red'],
            labels={'x': 'Taxa de Churn (%)', 'y': 'Cidade'}
        )
        fig_cidade.update_layout(height=350)
        st.plotly_chart(fig_cidade, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição de idade por churn
        st.markdown("**Distribuição de Idade por Churn**")
        fig_idade = px.histogram(
            clientes, 
            x='idade', 
            color='churn',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'churn': 'Cancelou', 'idade': 'Idade'},
            barmode='overlay',
            opacity=0.7
        )
        fig_idade.update_layout(height=350)
        st.plotly_chart(fig_idade, use_container_width=True)
    
    with col2:
        # Frequência de compras por churn
        if 'frequencia_compras' in clientes.columns:
            st.markdown("**Frequência de Compras por Churn**")
            fig_freq = px.box(
                clientes,
                x='churn',
                y='frequencia_compras',
                color='churn',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'churn': 'Cancelou', 'frequencia_compras': 'Número de Compras'}
            )
            fig_freq.update_layout(height=350)
            st.plotly_chart(fig_freq, use_container_width=True)
    
    # Matriz de correlação
    st.markdown("**Matriz de Correlação - Principais Features**")
    cols_corr = ['idade', 'recencia_dias', 'frequencia_compras', 'valor_total', 
                 'ticket_medio', 'churn']
    cols_existentes = [col for col in cols_corr if col in clientes.columns]
    
    if len(cols_existentes) > 1:
        corr_matrix = clientes[cols_existentes].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

# ============================================
# TAB 2 - MODELOS PREDITIVOS
# ============================================

with tab2:
    st.subheader("🤖 Comparação de Modelos Preditivos")
    
    if metricas is not None:
        # Gráfico de comparação
        st.markdown("**Comparação de Desempenho entre Modelos**")
        
        fig_compare = go.Figure()
        
        metricas_plot = ['Recall', 'AUC-ROC', 'F1-Score', 'Precisão']
        cores = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        for i, metrica in enumerate(metricas_plot):
            fig_compare.add_trace(go.Bar(
                name=metrica,
                x=metricas.index,
                y=metricas[metrica],
                marker_color=cores[i]
            ))
        
        fig_compare.update_layout(
            barmode='group',
            xaxis_title="Modelo",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=500,
            legend_title="Métricas"
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Tabela de métricas
        st.markdown("**Tabela de Métricas Detalhada**")
        st.dataframe(metricas.round(3), use_container_width=True)
        
        # Destaque para o melhor modelo
        melhor_recall = metricas['Recall'].idxmax()
        melhor_auc = metricas['AUC-ROC'].idxmax()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🏆 **Melhor Recall**: {melhor_recall} ({metricas.loc[melhor_recall, 'Recall']:.3f})")
        with col2:
            st.info(f"🏆 **Melhor AUC-ROC**: {melhor_auc} ({metricas.loc[melhor_auc, 'AUC-ROC']:.3f})")
        
        st.markdown("---")
        st.markdown("**📌 Conclusão:** O modelo **XGBoost Otimizado** foi escolhido como o melhor por priorizar **Recall** (identificar o máximo de clientes em risco), alcançando **77.1%** de recall.")
    else:
        st.warning("Dados de métricas não disponíveis")

# ============================================
# TAB 3 - SEGMENTAÇÃO DE RISCO
# ============================================

with tab3:
    st.subheader("📈 Segmentação de Clientes por Risco de Churn")
    
    if 'segmento_detalhado' in clientes.columns and 'probabilidade_final' in clientes.columns:
        # Distribuição dos segmentos
        segmentos_count = clientes['segmento_detalhado'].value_counts().reset_index()
        segmentos_count.columns = ['Segmento', 'Quantidade']
        
        # Ordem correta dos segmentos
        ordem_segmentos = ['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Crítico']
        segmentos_count['Segmento'] = pd.Categorical(
            segmentos_count['Segmento'], 
            categories=ordem_segmentos, 
            ordered=True
        )
        segmentos_count = segmentos_count.sort_values('Segmento')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribuição dos Segmentos de Risco**")
            cores_segmentos = ['#2ecc71', '#f1c40f', '#f39c12', '#e67e22', '#e74c3c']
            fig_pie = px.pie(
                segmentos_count, 
                values='Quantidade', 
                names='Segmento',
                color='Segmento',
                color_discrete_sequence=cores_segmentos,
                hole=0.4
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("**Churn Real por Segmento**")
            churn_segmento = clientes.groupby('segmento_detalhado')['churn'].mean().reset_index()
            churn_segmento.columns = ['Segmento', 'Taxa Churn']
            churn_segmento['Taxa Churn'] = churn_segmento['Taxa Churn'] * 100
            churn_segmento['Segmento'] = pd.Categorical(
                churn_segmento['Segmento'], 
                categories=ordem_segmentos, 
                ordered=True
            )
            churn_segmento = churn_segmento.sort_values('Segmento')
            
            fig_churn_seg = px.bar(
                churn_segmento,
                x='Segmento',
                y='Taxa Churn',
                color='Segmento',
                color_discrete_sequence=cores_segmentos,
                text_auto='.1f',
                labels={'Taxa Churn': 'Taxa de Churn (%)'}
            )
            fig_churn_seg.update_layout(height=400, showlegend=False)
            fig_churn_seg.update_traces(textposition='outside')
            st.plotly_chart(fig_churn_seg, use_container_width=True)
        
        # Tabela detalhada
        st.markdown("**Detalhamento dos Segmentos**")
        detalhe_segmentos = clientes.groupby('segmento_detalhado').agg({
            'id_cliente': 'count',
            'churn': 'mean',
            'idade': 'mean',
            'frequencia_compras': 'mean',
            'valor_total': 'mean',
            'probabilidade_final': 'mean'
        }).round(2)
        
        detalhe_segmentos.columns = [
            'Total Clientes', 'Taxa Churn Real', 'Idade Média', 
            'Frequência Média', 'Valor Total Médio', 'Probabilidade Média'
        ]
        detalhe_segmentos['Taxa Churn Real'] = detalhe_segmentos['Taxa Churn Real'] * 100
        
        # Reordenar
        detalhe_segmentos = detalhe_segmentos.reindex(ordem_segmentos)
        
        st.dataframe(detalhe_segmentos, use_container_width=True)
        
    else:
        st.warning("Dados de segmentação não disponíveis. Execute o notebook de modelagem primeiro.")

# ============================================
# TAB 4 - IMPORTÂNCIA DE FEATURES
# ============================================

with tab4:
    st.subheader("🔑 Fatores Mais Importantes para Churn")
    
    if importancias is not None and not importancias.empty:
        # Top 15 features
        top_features = importancias.head(15)
        
        fig_imp = px.bar(
            top_features,
            x='importancia',
            y='feature',
            orientation='h',
            color='importancia',
            color_continuous_scale='viridis',
            labels={'importancia': 'Importância', 'feature': 'Feature'}
        )
        fig_imp.update_layout(height=600)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Interpretação
        st.markdown("**📌 Interpretação das Principais Features:**")
        
        interpretacoes = {
            'recencia_dias': "🕐 **Recência**: Quanto mais dias sem comprar, maior a probabilidade de churn",
            'frequencia_compras': "📦 **Frequência**: Clientes que compram pouco têm maior risco",
            'valor_total': "💰 **Valor Total**: Baixo valor gasto indica menor engajamento",
            'dias_entre_compras': "⏱️ **Intervalo**: Longos intervalos entre compras sinalizam desinteresse",
            'canal_aquisicao': "📢 **Canal**: Canais como Facebook apresentam maior risco",
            'cidade': "📍 **Localização**: Diferenças regionais impactam o comportamento",
            'cliente_sem_compra': "🆕 **Sem Compras**: Clientes que nunca compraram têm comportamento atípico",
            'ticket_medio': "💵 **Ticket Médio**: Valor médio reflete perfil do cliente"
        }
        
        for feature in top_features['feature'].head(5):
            encontrou = False
            for chave, interpretacao in interpretacoes.items():
                if chave in feature:
                    st.markdown(interpretacao)
                    encontrou = True
                    break
            if not encontrou:
                st.markdown(f"• **{feature}**: Influencia diretamente a probabilidade de churn")
    else:
        st.warning("Dados de importância de features não disponíveis")

# ============================================
# TAB 5 - RECOMENDAÇÕES
# ============================================

with tab5:
    st.subheader("💡 Recomendações de Negócio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Por Segmento de Risco
        
        **🟢 Muito Baixo Risco**
        - Manter engajamento com newsletter mensal
        - Oferecer descontos sazonais
        - Programa de indicação
        
        **🟡 Baixo Risco**
        - Programa de fidelidade com pontos
        - Ofertas personalizadas
        - Email marketing segmentado
        
        **🟠 Médio Risco**
        - Aumentar frequência de contato
        - Benefícios exclusivos
        - Pesquisa de satisfação
        
        **🔴 Alto Risco**
        - Contato proativo
        - Oferta de retenção personalizada
        - Análise de reclamações
        
        **⚫ Crítico**
        - 🚨 INTERVENÇÃO IMEDIATA
        - Oferta especial (desconto agressivo)
        - Entrevista de saída
        """)
    
    with col2:
        st.markdown("""
        ### 📅 Plano de Ação
        
        **🔴 Primeiros 30 dias**
        1. Contatar segmento CRÍTICO
        2. Oferta especial para ALTO risco
        3. Alertas automáticos
        
        **🟡 30-60 dias**
        1. Programa de fidelidade para MÉDIO
        2. Campanhas para BAIXO risco
        3. Analisar feedback
        
        **🟢 60-90 dias**
        1. Automatizar newsletter
        2. Programa de indicação
        3. Dashboard contínuo
        """)
    
    st.markdown("---")
    
    # Impacto financeiro
    st.subheader("💰 Impacto Financeiro Estimado")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Receita Retida Anual", "R$ 284.550,00", delta="+35%")
    with col2:
        st.metric("ROI Estimado", "2,93x", delta="+193%")
    with col3:
        st.metric("Redução do Churn", "5,42 p.p.", delta="-24,3%")
    with col4:
        st.metric("Clientes Recuperados", "~1.650", delta="+650")

# ============================================
# RODAPÉ
# ============================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Desenvolvido por <b>Raphael Oliveira Bomfim</b> | 
        <a href='https://www.linkedin.com/in/raphael-oliveira-bomfim/'>LinkedIn</a> | 
        <a href='https://github.com/raphaeloliveirabomfim'>GitHub</a></p>
        <p>Projeto de Portfólio - Análise e Predição de Churn para E-commerce</p>
    </div>
    """,
    unsafe_allow_html=True
)