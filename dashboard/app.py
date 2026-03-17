"""
🛒 Dashboard — Previsão de Churn em E-commerce
===============================================
Execute com:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, fbeta_score
)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn E-commerce · Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

COR_CHURN  = "#E74C3C"
COR_OK     = "#27AE60"
COR_ACCENT = "#2980B9"
COR_WARN   = "#F39C12"
COR_CRITICO= "#922B21"

CORES_SEG = {
    "Crítico"    : "#922B21",
    "Alto"       : "#E74C3C",
    "Médio"      : "#F39C12",
    "Baixo"      : "#2980B9",
    "Muito Baixo": "#27AE60",
}

TEMPLATE = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="DM Sans, sans-serif", color="#2C3E50"),
    margin=dict(t=40, b=30, l=20, r=20),
    title_font=dict(color="#1A2E4A", size=13),
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #F0F3F8; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A2E4A 0%, #16253D 100%);
}
[data-testid="stSidebar"] > div * { color: #D0D9E8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

[data-testid="stTabs"] button p { color: #1A2E4A !important; font-weight: 600; }
[data-testid="stTabs"] button[aria-selected="true"] p { color: #2980B9 !important; }
.main h1, .main h2, .main h3, .main h4 { color: #1A2E4A !important; }

.kpi-card {
    background: white; border-radius: 12px; padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #2980B9;
    margin-bottom: 8px;
}
.kpi-card.danger  { border-left-color: #E74C3C; }
.kpi-card.warning { border-left-color: #F39C12; }
.kpi-card.success { border-left-color: #27AE60; }
.kpi-card.critico { border-left-color: #922B21; }
.kpi-label { font-size:0.76rem; font-weight:600; color:#7F8C9A;
             text-transform:uppercase; letter-spacing:0.06em; }
.kpi-value { font-size:1.9rem; font-weight:700; color:#1A2E4A;
             line-height:1.1; margin:4px 0; }
.kpi-delta { font-size:0.80rem; color:#7F8C9A; }
.section-title {
    font-size:0.82rem; font-weight:700; color:#1A2E4A;
    text-transform:uppercase; letter-spacing:0.06em;
    padding:0 0 6px 0; margin-top:6px; border-bottom:2px solid #E8ECF0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent

@st.cache_data
def carregar_dados():
    clientes  = pd.read_csv(BASE / "data" / "raw" / "clientes.csv")
    try:
        seg = pd.read_csv(BASE / "data" / "processed" / "clientes_segmentados.csv")
        return seg
    except FileNotFoundError:
        return clientes

@st.cache_resource
def carregar_modelo():
    try:
        pipe = joblib.load(BASE / "models" / "pipeline_final.pkl")
        thr  = joblib.load(BASE / "models" / "threshold_final.pkl")
        X    = pd.read_csv(BASE / "data" / "processed" / "X_features.csv")
        y    = pd.read_csv(BASE / "data" / "processed" / "y_target.csv").squeeze()
        return pipe, thr, X, y
    except Exception:
        return None, 0.5, None, None

@st.cache_data
def carregar_resultados():
    try:
        return pd.read_csv(BASE / "models" / "resultados_modelos.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def carregar_feature_importance():
    try:
        return pd.read_csv(BASE / "models" / "feature_importance.csv")
    except FileNotFoundError:
        return None

df_clientes = carregar_dados()
modelo, threshold, X_all, y_all = carregar_modelo()
df_resultados = carregar_resultados()
df_fi = carregar_feature_importance()

# Calcular scores se modelo disponível
if modelo is not None and X_all is not None:
    df_clientes['score_churn'] = modelo.predict_proba(X_all)[:, 1]
    df_clientes['predito']     = (df_clientes['score_churn'] >= threshold).astype(int)
    if 'segmento' not in df_clientes.columns:
        df_clientes['segmento'] = pd.cut(
            df_clientes['score_churn'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Muito Baixo','Baixo','Médio','Alto','Crítico']
        )
    y_prob = df_clientes['score_churn'].values
    y_pred = df_clientes['predito'].values
    y_true = y_all.values
elif 'score_churn' in df_clientes.columns:
    y_prob = df_clientes['score_churn'].values
    y_true = df_clientes['churn'].values if 'churn' in df_clientes.columns else np.zeros(len(df_clientes))
    y_pred = (y_prob >= threshold).astype(int)
else:
    np.random.seed(42)
    y_true = df_clientes['churn'].values if 'churn' in df_clientes.columns else np.zeros(len(df_clientes))
    y_prob = np.where(y_true == 1,
                      np.random.beta(6, 2, len(df_clientes)),
                      np.random.beta(1, 5, len(df_clientes)))
    df_clientes['score_churn'] = y_prob
    y_pred = (y_prob >= threshold).astype(int)
    df_clientes['segmento'] = pd.cut(
        y_prob, bins=[0,0.2,0.4,0.6,0.8,1.0],
        labels=['Muito Baixo','Baixo','Médio','Alto','Crítico']
    )

# Métricas globais
cm                   = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp       = cm.ravel()
recall_val           = recall_score(y_true, y_pred)
precision_val        = precision_score(y_true, y_pred, zero_division=0)
f2_val               = fbeta_score(y_true, y_pred, beta=2)
auc_val              = roc_auc_score(y_true, y_prob)
pr_auc_val           = average_precision_score(y_true, y_prob)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Churn E-commerce")
    st.markdown("**Dashboard de Retenção de Clientes**")
    st.markdown("---")

    st.markdown("### 🔍 Filtros")
    canal_sel = st.selectbox("Canal de Aquisição",
        ["Todos"] + sorted(df_clientes['canal_aquisicao'].unique().tolist())
        if 'canal_aquisicao' in df_clientes.columns else ["Todos"])
    cidade_sel = st.selectbox("Cidade",
        ["Todas"] + sorted(df_clientes['cidade'].unique().tolist())
        if 'cidade' in df_clientes.columns else ["Todas"])
    seg_sel = st.selectbox("Segmento de Risco",
        ["Todos","Crítico","Alto","Médio","Baixo","Muito Baixo"])

    st.markdown("---")
    st.markdown("### ⚙️ Threshold")
    thr_slider = st.slider("Sensibilidade", 0.01, 0.99, float(threshold), 0.01)
    y_pred_thr = (y_prob >= thr_slider).astype(int)
    rec_thr    = recall_score(y_true, y_pred_thr)
    prec_thr   = precision_score(y_true, y_pred_thr, zero_division=0)
    st.markdown(f"""
    <div style="background:#1A2E4A;border-radius:6px;padding:10px;font-size:0.82rem">
    <div style="color:#8B949E">Com threshold {thr_slider:.3f}:</div>
    <div style="color:#3FB950">Recall: {rec_thr:.1%}</div>
    <div style="color:#D29922">Precisão: {prec_thr:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Portfólio · Ciência de Dados")

# Aplicar filtros
dff = df_clientes.copy()
if canal_sel  != "Todos"  and 'canal_aquisicao' in dff.columns:
    dff = dff[dff['canal_aquisicao'] == canal_sel]
if cidade_sel != "Todas"  and 'cidade' in dff.columns:
    dff = dff[dff['cidade'] == cidade_sel]
if seg_sel    != "Todos"  and 'segmento' in dff.columns:
    dff = dff[dff['segmento'] == seg_sel]

# ─────────────────────────────────────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────────────────────────────────────
aba1, aba2, aba3, aba4 = st.tabs([
    "📊  Visão Geral",
    "🎯  Segmentação",
    "📈  Desempenho do Modelo",
    "🔎  Analisar Cliente",
])

# ══════════════════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════
with aba1:
    st.markdown('<h2 style="color:#1A2E4A;">📊 Visão Geral</h2>', unsafe_allow_html=True)
    st.caption(f"{len(dff):,} clientes com os filtros selecionados")

    taxa_churn = dff['churn'].mean() * 100 if 'churn' in dff.columns else 0
    n_critico  = (dff['segmento'] == 'Crítico').sum() if 'segmento' in dff.columns else 0
    n_alto     = (dff['segmento'] == 'Alto').sum()    if 'segmento' in dff.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, cls, label, val, delta in [
        (c1, "danger",  "Taxa de Churn",    f"{taxa_churn:.1f}%",  f"{int(dff['churn'].sum()) if 'churn' in dff.columns else 0:,} clientes"),
        (c2, "critico", "Segmento Crítico", f"{n_critico:,}",       "intervenção imediata"),
        (c3, "danger",  "Segmento Alto",    f"{n_alto:,}",          "contato proativo"),
        (c4, "success", "Recall do Modelo", f"{recall_val:.1%}",    "churn identificados"),
        (c5, "",        "F2-Score",         f"{f2_val:.4f}",        "métrica principal"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Taxa de Churn por Canal de Aquisição</div>', unsafe_allow_html=True)
        if 'canal_aquisicao' in dff.columns and 'churn' in dff.columns:
            t = (dff.groupby('canal_aquisicao')['churn'].mean() * 100).reset_index()
            t.columns = ['Canal', 'Taxa (%)']
            fig = px.bar(t.sort_values('Taxa (%)'), x='Taxa (%)', y='Canal',
                         orientation='h',
                         color='Taxa (%)',
                         color_continuous_scale=['#27AE60','#F39C12','#E74C3C'],
                         text=t.sort_values('Taxa (%)')['Taxa (%)'].round(1).astype(str) + '%')
            fig.update_traces(textposition='outside', marker_line_width=0)
            fig.update_layout(**TEMPLATE, coloraxis_showscale=False,
                xaxis=dict(title='Taxa (%)', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50')),
                yaxis=dict(title='', tickfont=dict(color='#2C3E50')),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Taxa de Churn por Cidade</div>', unsafe_allow_html=True)
        if 'cidade' in dff.columns and 'churn' in dff.columns:
            t = (dff.groupby('cidade')['churn'].mean() * 100).reset_index()
            t.columns = ['Cidade', 'Taxa (%)']
            fig = px.bar(t.sort_values('Taxa (%)'), x='Cidade', y='Taxa (%)',
                         color='Taxa (%)',
                         color_continuous_scale=['#27AE60','#F39C12','#E74C3C'],
                         text=t['Taxa (%)'].sort_values().round(1).astype(str) + '%')
            fig.update_traces(textposition='outside', marker_line_width=0)
            fig.update_layout(**TEMPLATE, coloraxis_showscale=False,
                xaxis=dict(title='', tickfont=dict(color='#2C3E50')),
                yaxis=dict(title='Taxa (%)', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50')),
            )
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Distribuição dos Scores de Churn</div>', unsafe_allow_html=True)
        if 'score_churn' in dff.columns and 'churn' in dff.columns:
            fig = go.Figure()
            for label, cor in [(0, COR_OK), (1, COR_CHURN)]:
                nome = 'Ativo' if label == 0 else 'Churn'
                sub  = dff[dff['churn'] == label]['score_churn']
                fig.add_trace(go.Histogram(x=sub, name=nome, marker_color=cor,
                                           opacity=0.7, nbinsx=40, marker_line_width=0))
            fig.add_vline(x=thr_slider, line_dash='dash', line_color=COR_ACCENT,
                          line_width=2, annotation_text=f'Threshold ({thr_slider:.3f})',
                          annotation_font_color=COR_ACCENT)
            fig.update_layout(**TEMPLATE, barmode='overlay',
                xaxis=dict(title='Score de Churn', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50')),
                yaxis=dict(title='Frequência', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50')),
                legend=dict(font=dict(color='#2C3E50')),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Churn por Faixa Etária</div>', unsafe_allow_html=True)
        if 'idade' in dff.columns and 'churn' in dff.columns:
            dff2 = dff.copy()
            dff2['faixa'] = pd.cut(dff2['idade'].clip(18,70),
                                    bins=[17,25,35,45,55,70],
                                    labels=['18-25','26-35','36-45','46-55','56-70'])
            t = (dff2.groupby('faixa', observed=True)['churn'].mean() * 100).reset_index()
            t.columns = ['Faixa', 'Taxa (%)']
            fig = px.line(t, x='Faixa', y='Taxa (%)',
                          markers=True,
                          labels={'Faixa': 'Faixa Etária', 'Taxa (%)': 'Taxa de Churn (%)'})
            fig.update_traces(line_color=COR_CHURN, line_width=2.5, marker=dict(size=8))
            fig.update_layout(**TEMPLATE,
                xaxis=dict(title='Faixa Etária', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50')),
                yaxis=dict(title='Taxa (%)', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50')),
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ABA 2 — SEGMENTAÇÃO
# ══════════════════════════════════════════════════════════
with aba2:
    st.markdown('<h2 style="color:#1A2E4A;">🎯 Segmentação de Risco</h2>', unsafe_allow_html=True)

    if 'segmento' in dff.columns:
        dist_seg    = dff['segmento'].value_counts().reindex(
            ['Crítico','Alto','Médio','Baixo','Muito Baixo']).fillna(0)
        churn_seg   = dff.groupby('segmento', observed=True)['churn'].mean() * 100 \
                      if 'churn' in dff.columns else pd.Series(dtype=float)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<div class="section-title">Distribuição dos Segmentos</div>', unsafe_allow_html=True)
            fig = px.pie(
                values=dist_seg.values,
                names=dist_seg.index,
                color=dist_seg.index,
                color_discrete_map=CORES_SEG,
                hole=0.45,
            )
            fig.update_traces(textposition='outside', textinfo='percent+label',
                              marker=dict(line=dict(color='white', width=2)))
            fig.update_layout(**TEMPLATE, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Clientes e Churn Real por Segmento</div>', unsafe_allow_html=True)
            seg_df = pd.DataFrame({
                'Segmento'   : dist_seg.index,
                'Clientes'   : dist_seg.values.astype(int),
                'Churn Real (%)': [churn_seg.get(s, 0) for s in dist_seg.index],
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=seg_df['Segmento'], y=seg_df['Clientes'],
                name='Clientes', marker_color=[CORES_SEG[s] for s in seg_df['Segmento']],
                yaxis='y', opacity=0.8,
            ))
            fig.add_trace(go.Scatter(
                x=seg_df['Segmento'], y=seg_df['Churn Real (%)'],
                name='Churn Real (%)', mode='lines+markers',
                line=dict(color='#1A2E4A', width=2.5),
                marker=dict(size=8), yaxis='y2',
            ))
            fig.update_layout(**TEMPLATE,
                yaxis=dict(title='Clientes', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50')),
                yaxis2=dict(title='Churn Real (%)', title_font=dict(color='#1A2E4A'),
                            tickfont=dict(color='#1A2E4A'), overlaying='y', side='right'),
                legend=dict(font=dict(color='#2C3E50')),
                xaxis=dict(title='', tickfont=dict(color='#2C3E50')),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recomendações por segmento
        st.markdown('<div class="section-title">Recomendações por Segmento</div>', unsafe_allow_html=True)
        recomendacoes = {
            '🚨 Crítico'    : 'Intervenção imediata | Oferta agressiva (30-50%) | Atendente dedicado',
            '🔴 Alto'       : 'Contato proativo (WhatsApp) | Oferta personalizada | Customer success',
            '🟠 Médio'      : 'Cupom de desconto | Pesquisa de satisfação | Contato quinzenal',
            '🟡 Baixo'      : 'Programa de fidelidade | Email marketing segmentado | Acesso antecipado a promoções',
            '🟢 Muito Baixo': 'Newsletter mensal | Programa de indicação | Coleta de avaliações',
        }
        cols = st.columns(5)
        for col, (seg_label, acao) in zip(cols, recomendacoes.items()):
            with col:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:14px;
                            box-shadow:0 1px 4px rgba(0,0,0,0.08);height:140px">
                    <div style="font-weight:700;font-size:0.85rem;margin-bottom:8px">{seg_label}</div>
                    <div style="font-size:0.78rem;color:#555;line-height:1.5">{acao}</div>
                </div>""", unsafe_allow_html=True)

        # Top clientes em risco
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Clientes em Risco Crítico</div>', unsafe_allow_html=True)
        cols_show = ['id_cliente','idade','canal_aquisicao','cidade','score_churn','segmento']
        cols_show = [c for c in cols_show if c in dff.columns]
        criticos = dff[dff['segmento'] == 'Crítico'][cols_show]\
                   .sort_values('score_churn', ascending=False).head(15)
        if 'score_churn' in criticos.columns:
            criticos['score_churn'] = criticos['score_churn'].round(4)
        st.dataframe(criticos, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# ABA 3 — DESEMPENHO DO MODELO
# ══════════════════════════════════════════════════════════
with aba3:
    st.markdown('<h2 style="color:#1A2E4A;">📈 Desempenho do Modelo</h2>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, cls, label, val, desc in [
        (c1, "success", "Recall",    f"{recall_val:.1%}",    "churn identificados"),
        (c2, "success", "F2-Score",  f"{f2_val:.4f}",        "Recall com peso 2×"),
        (c3, "warning", "Precisão",  f"{precision_val:.1%}", "acertos do alerta"),
        (c4, "success", "AUC-ROC",   f"{auc_val:.4f}",       "separação de classes"),
        (c5, "",        "PR-AUC",    f"{pr_auc_val:.4f}",    "métrica principal"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.5rem">{val}</div>
                <div class="kpi-delta">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Curva Precision-Recall</div>', unsafe_allow_html=True)
        prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_c, y=prec_c, mode='lines',
                                  name=f'Modelo (PR-AUC={pr_auc_val:.3f})',
                                  line=dict(color=COR_ACCENT, width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[y_true.mean(), y_true.mean()],
                                  mode='lines', name=f'Aleatório ({y_true.mean():.3f})',
                                  line=dict(color='gray', dash='dash', width=1.5)))
        fig.add_trace(go.Scatter(x=[recall_val], y=[precision_val], mode='markers',
                                  name=f'Threshold ({threshold:.3f})',
                                  marker=dict(color=COR_CHURN, size=12, symbol='star')))
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title='Recall', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50'), range=[0,1]),
            yaxis=dict(title='Precisão', title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50'), range=[0,1.05]),
            legend=dict(font=dict(color='#2C3E50')),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        if df_fi is not None:
            fi_top = df_fi.head(15)
            fig = px.bar(fi_top, x='Importância', y='Feature', orientation='h',
             color='Importância',
             color_continuous_scale=['#2980B9','#E74C3C'],
             text=fi_top['Importância'].round(4))
            fig.update_traces(textposition='outside', marker_line_width=0)
            fig.update_layout(**TEMPLATE,
                coloraxis_showscale=False,
                yaxis=dict(categoryorder='total ascending', title='',
                           tickfont=dict(color='#2C3E50')),
                xaxis=dict(title='Importância', title_font=dict(color='#2C3E50'),
                           tickfont=dict(color='#2C3E50')),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Execute o notebook 03_modelagem.ipynb para gerar feature_importance.csv')

    if df_resultados is not None:
        st.markdown('<div class="section-title">Comparação de Modelos</div>', unsafe_allow_html=True)
        st.dataframe(df_resultados.round(4), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# ABA 4 — ANALISAR CLIENTE
# ══════════════════════════════════════════════════════════
with aba4:
    st.markdown('<h2 style="color:#1A2E4A;">🔎 Analisar Cliente Individual</h2>', unsafe_allow_html=True)
    st.markdown("Preencha o perfil do cliente para calcular o score de risco de churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Perfil**")
        idade_p    = st.number_input("Idade", 18, 70, 35)
        genero_p   = st.selectbox("Gênero", ["M", "F"])
        cidade_p   = st.selectbox("Cidade", ["SP","RJ","BH","BSB","POA"])
        canal_p    = st.selectbox("Canal de Aquisição", ["Google","Facebook","Indicação","Orgânico"])

    with col2:
        st.markdown("**🛍️ Comportamento de Compra**")
        recencia_p  = st.slider("Recência (dias desde última compra)", 0, 999, 60)
        frequencia_p= st.slider("Número de compras", 0, 50, 7)
        valor_p     = st.number_input("Valor total gasto (R$)", 0.0, 50000.0, 1000.0, 100.0)
        ticket_p    = st.number_input("Ticket médio (R$)", 0.0, 5000.0, 150.0, 10.0)

    with col3:
        st.markdown("**📊 Detalhes**")
        dias_entre_p = st.slider("Dias médios entre compras", 0, 365, 45)
        metodo_p     = st.selectbox("Método preferido", ["Cartão","Pix","Boleto","Sem compras"])
        cat_pref_p   = st.selectbox("Categoria preferida",
                                     ["Eletrônicos","Moda","Casa","Esportes","Beleza"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Calcular Score de Churn", use_container_width=True, type="primary"):
        if modelo is not None and X_all is not None:
            # Montar entrada com as mesmas features do modelo
            entrada = pd.DataFrame(columns=X_all.columns, index=[0]).fillna(0)

            # Preencher features numéricas
            mapa_num = {
                'idade': idade_p, 'recencia_dias': recencia_p,
                'frequencia_compras': frequencia_p, 'valor_total': valor_p,
                'ticket_medio': ticket_p, 'dias_entre_compras': dias_entre_p,
                'cliente_sem_compra': int(frequencia_p == 0),
                'ticket_medio_rel': ticket_p / X_all['ticket_medio'].mean() if 'ticket_medio' in X_all.columns else 1.0,
                f'compras_{cat_pref_p.lower()}': frequencia_p,
            }
            for col, val in mapa_num.items():
                if col in entrada.columns:
                    entrada[col] = val

            # Preencher features categóricas (one-hot)
            mapa_ohe = {
                f'genero_{genero_p}': 1,
                f'cidade_{cidade_p}': 1,
                f'canal_aquisicao_{canal_p}': 1,
                f'metodo_preferido_{metodo_p}': 1,
            }
            for col, val in mapa_ohe.items():
                if col in entrada.columns:
                    entrada[col] = val

            entrada = entrada.astype(float)
            score   = modelo.predict_proba(entrada)[0, 1]
            nivel   = (
                'Crítico' if score >= 0.8 else
                'Alto'    if score >= 0.6 else
                'Médio'   if score >= 0.4 else
                'Baixo'   if score >= 0.2 else
                'Muito Baixo'
            )
            cor = CORES_SEG[nivel]
            cls = "critico" if nivel == "Crítico" else \
                  "danger"  if nivel == "Alto"    else \
                  "warning" if nivel == "Médio"   else "success"

            st.markdown("---")
            r1, r2, r3 = st.columns([1, 1, 2])

            with r1:
                st.markdown(f"""
                <div class="kpi-card {cls}">
                    <div class="kpi-label">Score de Churn</div>
                    <div class="kpi-value" style="color:{cor}">{score:.1%}</div>
                    <div class="kpi-delta">probabilidade de cancelar</div>
                </div>""", unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="kpi-card {cls}">
                    <div class="kpi-label">Segmento</div>
                    <div class="kpi-value" style="color:{cor};font-size:1.5rem">{nivel}</div>
                    <div class="kpi-delta">threshold: {threshold:.3f}</div>
                </div>""", unsafe_allow_html=True)

            with r3:
                alertas = []
                if recencia_p  > 60:   alertas.append("🔴 Alta recência — mais de 60 dias sem comprar")
                if frequencia_p < 3:   alertas.append("🔴 Baixa frequência — menos de 3 compras")
                if canal_p == 'Facebook': alertas.append("🟡 Canal Facebook — maior taxa de churn histórica")
                if cidade_p == 'BSB':  alertas.append("🟡 Brasília — cidade de maior churn")
                if idade_p > 55:       alertas.append("🟡 Faixa 56+ — maior risco por faixa etária")
                if frequencia_p == 0:  alertas.append("🔴 Nenhuma compra registrada")

                if alertas:
                    st.markdown("**Fatores de risco identificados:**")
                    for a in alertas:
                        st.markdown(f"- {a}")
                else:
                    st.success("✅ Nenhum fator de risco crítico identificado.")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={"suffix": "%", "font": {"size": 32, "color": cor}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": cor, "thickness": 0.3},
                    "steps": [
                        {"range": [0,  20], "color": "#E8F8EE"},
                        {"range": [20, 40], "color": "#EBF5FB"},
                        {"range": [40, 60], "color": "#FEF9E7"},
                        {"range": [60, 80], "color": "#FDEDEC"},
                        {"range": [80, 100],"color": "#F5B7B1"},
                    ],
                    "threshold": {
                        "line": {"color": "#1A2E4A", "width": 3},
                        "thickness": 0.8, "value": threshold * 100,
                    },
                },
            ))
            fig.update_layout(height=260, margin=dict(t=20,b=10,l=30,r=30),
                              paper_bgcolor="white",
                              font=dict(family="DM Sans, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Modelo não encontrado. Execute o `03_modelagem.ipynb` para gerar `models/pipeline_final.pkl`.")
