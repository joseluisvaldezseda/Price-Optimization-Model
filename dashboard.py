import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Intelligence Dashboard | Quintiles",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS STYLE (Aesthetic & Minimalist)
st.markdown("""
    <style>
    .main { background-color: #F8F9FB; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #E5E7EB;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 8px;
        border-bottom: 1.5px solid #F3F4F6;
    }
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #3B82F6 !important;
        border-radius: 4px !important;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. OPTIMIZED DATA LOADING
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('rfm_churn_ltv.parquet')
        
        df.columns = df.columns.str.strip()
        df = df.replace('Dato no disponible', pd.NA)
        
        numeric_cols = ['Probabilidad_Churn', 'Probabilidad_Compra_90d', 'Monto_Esperado_90d', 'CLV_90dias']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- ROBUST SEGMENTATION ---
        
        # 1. Repurchase Propensity
        p_col = 'Probabilidad_Compra_90d'
        if p_col in df.columns and df[p_col].notnull().any():
            bins_p = [-0.001, 0.25, 0.50, 0.75, 1.001]
            labels_p = ['Low Propensity', 'Medium Propensity', 'High Propensity', 'Very High Propensity']
            df['Segmento_Recompra'] = pd.cut(df[p_col], bins=bins_p, labels=labels_p)
        else:
            df['Segmento_Recompra'] = 'No data'

        # 2. Future Value (Quartiles)
        v_col = 'CLV_90dias'
        if v_col in df.columns and df[v_col].notnull().any():
            try:
                quantile_labels = ['Bronze Value', 'Silver Value', 'Gold Value', 'Diamond Value']
                edges = df[v_col].quantile([0, 0.25, 0.5, 0.75, 1.0]).unique()
                if len(edges) > 1:
                    actual_labels = quantile_labels[:len(edges)-1]
                    df['Potencial_Valor'] = pd.cut(df[v_col], bins=edges, labels=actual_labels, include_lowest=True)
                else:
                    df['Potencial_Valor'] = 'Unique Value'
            except:
                df['Potencial_Valor'] = 'Undetermined'
        else:
            df['Potencial_Valor'] = 'No data'
            
        return df
    except Exception as e:
        st.error(f"Critical error loading file: {e}")
        return None

df_raw = load_data()

# --- 4. SIDEBAR (FILTERS) ---
if df_raw is not None:
    # Defining RFM ranking order
    segmento_a_calificacion = {
        'Campeones': 1, 'VIPs Leales': 2, 'Alto Potencial': 3, 'VIPs Potenciales': 4,
        'Calidad Reciente': 5, 'Nuevos Grandes Compradores': 6, 'Calidad Regular': 7,
        'Estándar Reciente': 8, 'Estándar': 9, 'Calidad Prometedora': 10,
        'Alto Riesgo - Valiosos': 11, 'Nuevos Clientes': 12, 'Bajo Compromiso': 13,
        'Pasivos': 14, 'Críticos a Retener': 15, 'Necesitan Atención': 16,
        'En Riesgo': 17, 'Baja Prioridad': 18, 'A Punto de Dormir': 19,
        'Hibernando': 20
    }

    with st.sidebar:
        st.markdown('<p class="sidebar-title">🎯 CRM Strategy</p>', unsafe_allow_html=True)
        
        explicaciones = {
            "Manual / All": "Full filter control. No automatic pre-selections.",
            "Golden Shield (VIP Retention)": "🛡️ <b>Objective:</b> Prevent value churn. VIP clients at high risk.",
            "Express Checkout (Conversion)": "⚡ <b>Objective:</b> Cash flow. Clients with high purchase propensity.",
            "Rough Diamonds (Upselling)": "💎 <b>Objective:</b> Account growth. New clients with high future CLV.",
            "Operation Lazarus (Reactivation)": "🔄 <b>Objective:</b> Recover accounts. Valuable inactive clients."
        }

        # 1. Database base lists
        all_bus = sorted(df_raw['NEGOCIO'].dropna().unique().tolist()) if 'NEGOCIO' in df_raw.columns else []
        all_rec = sorted(df_raw['Segmento_Recompra'].dropna().unique().tolist())
        all_val = sorted(df_raw['Potencial_Valor'].dropna().unique().tolist())
        all_rfm = sorted(df_raw['Segmento_RFM'].dropna().unique().tolist(), key=lambda x: segmento_a_calificacion.get(x, 99))
        all_risk = sorted(df_raw['Categoria_Probabilidad_Abandono'].dropna().unique().tolist())

        # 2. Strategy Callback
        def aplicar_estrategia():
            est = st.session_state.selector_estrategia
            st.session_state.ms_bus = all_bus
            st.session_state.ms_rec = all_rec
            st.session_state.ms_val = all_val
            st.session_state.ms_rfm = all_rfm
            st.session_state.ms_risk = all_risk

            if est == "Golden Shield (VIP Retention)":
                st.session_state.ms_rfm = [s for s in all_rfm if s in ['Campeones', 'VIPs Leales', 'Alto Potencial']]
                st.session_state.ms_risk = [s for s in all_risk if 'Alta' in s or 'Muy alta' in s]
                st.session_state.ms_val = [s for s in all_val if 'Gold' in s or 'Diamond' in s]
            elif est == "Express Checkout (Conversion)":
                st.session_state.ms_rec = [s for s in all_rec if 'High' in s or 'Very High' in s]
                st.session_state.ms_rfm = [s for s in all_rfm if s in ['Calidad Reciente', 'Nuevos Grandes Compradores', 'Estándar Reciente']]
            elif est == "Rough Diamonds (Upselling)":
                st.session_state.ms_rfm = [s for s in all_rfm if s in ['Nuevos Clientes', 'Calidad Prometedora', 'Estándar']]
                st.session_state.ms_val = [s for s in all_val if 'Gold' in s or 'Diamond' in s]
            elif est == "Operation Lazarus (Reactivation)":
                st.session_state.ms_rfm = [s for s in all_rfm if s in ['Críticos a Retener', 'En Riesgo', 'A Punto de Dormir']]
                st.session_state.ms_val = [s for s in all_val if 'Gold' in s or 'Silver' in s]

        def toggle_select_all(key_checkbox, key_ms, full_list):
            if st.session_state[key_checkbox]:
                st.session_state[key_ms] = full_list
            else:
                st.session_state[key_ms] = []

        # 3. Strategy Selector
        estrategia = st.selectbox(
            "Select a campaign preset:",
            list(explicaciones.keys()),
            key="selector_estrategia",
            on_change=aplicar_estrategia,
            help="Choosing a strategy will automatically adjust the filters below."
        )
        
        st.markdown(f"""<div style="background-color: #F9FAFB; border: 1px solid #E5E7EB; padding: 12px; border-radius: 8px; font-size: 0.82rem; color: #6B7280; line-height: 1.4; margin-bottom: 20px;">{explicaciones[estrategia]}</div>""", unsafe_allow_html=True)
        
        st.divider()
        st.markdown('<p class="sidebar-title">Filter Configuration</p>', unsafe_allow_html=True)

        # 4. FILTER RENDERING
        if 'NEGOCIO' in df_raw.columns:
            with st.expander("Business Type", expanded=False):
                st.checkbox("Select all", key="chk_bus", value=True, on_change=toggle_select_all, args=("chk_bus", "ms_bus", all_bus))
                if "ms_bus" not in st.session_state: st.session_state.ms_bus = all_bus
                selected_bus = st.multiselect("Business", all_bus, key="ms_bus", label_visibility="collapsed")
        
        with st.expander("Repurchase Propensity", expanded=False):
            st.checkbox("Select all", key="chk_rec", value=True, on_change=toggle_select_all, args=("chk_rec", "ms_rec", all_rec))
            if "ms_rec" not in st.session_state: st.session_state.ms_rec = all_rec
            selected_recompra = st.multiselect("Repurchase", all_rec, key="ms_rec", label_visibility="collapsed")

        with st.expander("Future Value Level", expanded=False):
            st.checkbox("Select all", key="chk_val", value=True, on_change=toggle_select_all, args=("chk_val", "ms_val", all_val))
            if "ms_val" not in st.session_state: st.session_state.ms_val = all_val
            selected_valor = st.multiselect("Value", all_val, key="ms_val", label_visibility="collapsed")

        with st.expander("RFM Segments", expanded=False):
            st.checkbox("Select all", key="chk_rfm", value=True, on_change=toggle_select_all, args=("chk_rfm", "ms_rfm", all_rfm))
            if "ms_rfm" not in st.session_state: st.session_state.ms_rfm = all_rfm
            selected_rfm = st.multiselect("RFM", all_rfm, key="ms_rfm", label_visibility="collapsed")
        
        with st.expander("Risk Levels", expanded=False):
            st.checkbox("Select all", key="chk_risk", value=True, on_change=toggle_select_all, args=("chk_risk", "ms_risk", all_risk))
            if "ms_risk" not in st.session_state: st.session_state.ms_risk = all_risk
            selected_abandon = st.multiselect("Risk", all_risk, key="ms_risk", label_visibility="collapsed")

        if st.button("Clear Filters / Reset"):
            for k in st.session_state.keys():
                if k.startswith("ms_") or k.startswith("chk_") or k == "selector_estrategia":
                    del st.session_state[k]
            st.rerun()


    # --- FILTERING ---
    mask = (
        (df_raw['Segmento_RFM'].isin(selected_rfm)) &
        (df_raw['Categoria_Probabilidad_Abandono'].isin(selected_abandon)) &
        (df_raw['Segmento_Recompra'].isin(selected_recompra)) &
        (df_raw['Potencial_Valor'].isin(selected_valor))
    )
    if 'NEGOCIO' in df_raw.columns:
        mask = mask & (df_raw['NEGOCIO'].isin(selected_bus))
        
    df_filtered = df_raw[mask].copy()


    # --- 5. MAIN BODY ---
    st.title("Business Insights")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Selected Accounts", f"{len(df_filtered):,}")
    with m2:
        val = df_filtered['Probabilidad_Churn'].mean()
        st.metric("Avg Churn Risk", f"{val:.1%}" if pd.notnull(val) else "0%")
    with m3:
        val = df_filtered['CLV_90dias'].sum()
        st.metric("Total CLV (90d)", f"${val:,.0f}")
    with m4:
        val = df_filtered['Probabilidad_Compra_90d'].mean()
        st.metric("Purchase Propensity", f"{val:.1%}" if pd.notnull(val) else "0%")

    # --- 6. CHARTS ---
    c1, c2 = st.columns(2)
    
    with c1:
        df_rfm_counts = df_filtered['Segmento_RFM'].value_counts().reset_index()
        df_rfm_counts.columns = ['Segment', 'Count']
        
        fig_rfm = px.bar(
            df_rfm_counts, 
            x='Count', 
            y='Segment', 
            orientation='h',
            title="<b>RFM Segment Distribution</b>",
            color='Count',
            color_continuous_scale='Sunsetdark', 
            text_auto='.2s' 
        )
        
        fig_rfm.update_traces(
            marker_line_width=0,
            opacity=0.9,
            textposition='outside',
            textfont=dict(size=12, color='#4B5563')
        )
        
        fig_rfm.update_layout(
            yaxis=dict(
                categoryorder='total ascending', 
                showgrid=False, 
                title=None
            ),
            font=dict(family="Inter, sans-serif", size=13, color="#1F2937"),
            title_font_size=15,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True, 
                gridcolor='#F3F4F6',
                showline=False,
                zeroline=False,
                title="Accounts"
            ),
            coloraxis_showscale=False, 
            height=550,
            margin=dict(l=20, r=40, t=70, b=20)
        )
        st.plotly_chart(fig_rfm, use_container_width=True)


    with c2:
        fig_v = px.pie(
            df_filtered, 
            names='Potencial_Valor', 
            title="<b>Accounts by Value Level (CLV)</b>",
            hole=0.5, 
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig_v.update_layout(
            font=dict(family="Inter, sans-serif", size=13, color="#1F2937"),
            title_font_size=15,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=550, 
            margin=dict(l=20, r=20, t=70, b=20)
        )
        st.plotly_chart(fig_v, use_container_width=True)


    # ROW 2: Predictive Analysis
    st.markdown('<p class="section-header">Propensity vs. Expected Value Relationship</p>', unsafe_allow_html=True)
    
    df_scat = df_filtered.dropna(subset=['Probabilidad_Compra_90d', 'Monto_Esperado_90d']).head(3000)
    
    if not df_scat.empty:
        fig_s = px.scatter(
            df_scat, 
            x='Probabilidad_Compra_90d', 
            y='Monto_Esperado_90d',
            size='CLV_90dias', 
            color='Segmento_Recompra', 
            title="",
            color_discrete_sequence=px.colors.qualitative.Safe,
            hover_name='CUENTA',
            hover_data={
                'Segmento_RFM': True,
                'Categoria_Probabilidad_Abandono': True,
                'Probabilidad_Compra_90d': ':.2%',
                'Monto_Esperado_90d': ':$,.2f',
                'CLV_90dias': ':$,.2f',
                'Segmento_Recompra': False
            }
        )
        
        fig_s.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Inter"
            ),
            xaxis_title="Purchase Probability (%)",
            yaxis_title="Expected Amount ($)"
        )
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("Insufficient predictive data for the scatter plot.")


    # --- 7. TABLE ---
    st.markdown('<p class="section-header">Customer Explorer & Recommendations</p>', unsafe_allow_html=True)
    
    search_query = st.text_input("Search specific account:", placeholder="Enter Account ID...")
    
    df_table = df_filtered.copy()
    if search_query:
        df_table = df_table[df_table['CUENTA'].astype(str).str.contains(search_query, case=False, na=False)]

    cols_t = [
        'CUENTA', 
        'NEGOCIO', 
        'Segmento_RFM', 
        'Potencial_Valor', 
        'CLV_90dias', 
        'Producto_Recomendado', 
        'Confianza_Recomendacion',
        'Categoria_Cross_Sell'
    ]

    st.dataframe(
        df_table[cols_t], use_container_width=True, height=400,
        column_config={
            "CLV_90dias": st.column_config.NumberColumn("Projected Value", format="$%.2f"),
            "CUENTA": st.column_config.TextColumn("Account"),
            "NEGOCIO": "Business",
            "Segmento_RFM": "RFM Segment",
            "Potencial_Valor": "Value Potential",
            "Producto_Recomendado": "Recommended Product",
            "Confianza_Recomendacion": "Confidence",
            "Categoria_Cross_Sell": "Cross Sell Category"
        }, hide_index=True
    )

    # --- 8. INSIGHTS ---
    st.markdown('<p class="section-header">Executive Summary</p>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        count = len(df_filtered[df_filtered['Potencial_Valor'].astype(str).str.contains('Diamond|Gold', na=False)])
        st.success(f"**VIP Accounts:**\n\n{count:,} accounts are in the top tier of projected value.")
    with i2:
        count = len(df_filtered[df_filtered['Segmento_Recompra'].astype(str).str.contains('Very High|High', na=False)])
        st.info(f"**Upcoming Sales:**\n\n{count:,} accounts have high or very high purchase propensity.")
    with i3:
        count = len(df_filtered[df_filtered['Categoria_Probabilidad_Abandono'].astype(str).str.contains('Alta', na=False)])
        st.error(f"**Critical Risk:**\n\n{count:,} accounts require immediate attention due to churn risk.")
else:
    st.error("File not found or empty.")