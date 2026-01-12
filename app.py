import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Configuración de página
st.set_page_config(page_title="Pricing Optimization Tool", layout="wide")

# --- FUNCIONES DE CARGA Y PROCESAMIENTO ---
@st.cache_data
def load_and_clean_data(file_path):
    # Carga desde Parquet como solicitaste
    df = pd.read_parquet("Base pricing NEW CSV.parquet")
    
    renaming_dict = {
        "Año": "Year", "Semana": "Week", "Departamento": "Department",
        "Material": "Product_ID", "Descripción de Material": "Product_Description",
        "Marca": "Brand", "Grupo de Artículo": "Product_Group",
        "Venta Costo": "Sales_Cost", "Venta Pzs": "Units_Sold", "Venta": "Total_Sales"
    }
    df.rename(columns=renaming_dict, inplace=True)
    
    cols_to_drop = ["Inv cto", "Inv pzas", "Precio SAP"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    return df

def process_product(df_product):
    df_agg = df_product.groupby(["Year", "Week"], as_index=False).agg({
        "Total_Sales": "sum",
        "Units_Sold": "sum",
        "Product_Description": "first",
        "Brand": "first"
    })
    
    df_agg["week_sin"] = np.sin(2 * np.pi * df_agg["Week"] / 52)
    df_agg["week_cos"] = np.cos(2 * np.pi * df_agg["Week"] / 52)
    
    df_agg["price"] = df_agg["Total_Sales"] / df_agg["Units_Sold"].replace(0, np.nan)
    df_agg["price"] = df_agg["price"].fillna(df_agg["price"].mean())
    
    return df_agg

# --- INTERFAZ DE USUARIO ---
st.title("Optimizador de Precios")

try:
    df_raw = load_and_clean_data("Base pricing NEW CSV.parquet")
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Filtros de Análisis")
min_rows = st.sidebar.slider("Mínimo de semanas de historial", 10, 150, 130)
seuil_corr = st.sidebar.slider("Umbral de correlación (Elasticidad)", 0.0, 1.0, 0.4)

product_counts = df_raw["Product_ID"].value_counts()
valid_products = product_counts[product_counts >= min_rows].index.tolist()
df_filtered = df_raw[df_raw["Product_ID"].isin(valid_products)]

if not valid_products:
    st.warning(f"No hay productos con más de {min_rows} semanas.")
    st.stop()

selected_id = st.selectbox("Selecciona un Producto (Product_ID)", valid_products)
product_info = df_filtered[df_filtered["Product_ID"] == selected_id].iloc[0]
st.subheader(f"{product_info['Product_Description']} | Marca: {product_info['Brand']}")

df_p = process_product(df_filtered[df_filtered["Product_ID"] == selected_id])

# Métricas
corr = df_p["price"].corr(df_p["Units_Sold"])
col1, col2, col3, col4 = st.columns(4)
col1.metric("Semanas de datos", len(df_p))
col2.metric("Elasticidad (Corr)", f"{corr:.2f}")
col3.metric("Precio Promedio", f"${df_p['price'].mean():,.2f}")
col4.metric("Ventas Totales (Pzs)", f"{df_p['Units_Sold'].sum():,.0f}")

# --- MODELADO Y OPTIMIZACIÓN ---
st.divider()
st.subheader("Simulación de Optimización")

X = df_p[["price", "week_sin", "week_cos"]]
y = df_p["Units_Sold"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

best_r2, best_model, best_name = -np.inf, None, ""
for name, model in models.items():
    model.fit(X_scaled, y)
    r2 = r2_score(y, model.predict(X_scaled))
    if r2 > best_r2:
        best_r2, best_model, best_name = r2, model, name

st.write(f"Mejor modelo: **{best_name}** (R²: {best_r2:.3f})")

# Simulación
prix_range = np.linspace(df_p["price"].min() * 0.8, df_p["price"].max() * 1.2, 50)
next_week = (df_p["Week"].max() + 1) % 52
simulacion = []

for p in prix_range:
    X_opt = pd.DataFrame({
        "price": [p],
        "week_sin": [np.sin(2 * np.pi * next_week / 52)],
        "week_cos": [np.cos(2 * np.pi * next_week / 52)]
    })
    X_opt_scaled = scaler.transform(X_opt)
    units_pred = max(0, best_model.predict(X_opt_scaled)[0])
    simulacion.append({"Precio": p, "Ventas": units_pred, "Ingreso": p * units_pred})

df_sim = pd.DataFrame(simulacion)
opt_row = df_sim.loc[df_sim["Ingreso"].idxmax()]

col_a, col_b = st.columns(2)
with col_a:
    st.success(f"**Precio Óptimo: ${opt_row['Precio']:,.2f}**")
with col_b:
    st.info(f"**Ingreso Estimado: ${opt_row['Ingreso']:,.2f}**")

# --- VISUALIZACIONES MEJORADAS ---
st.divider()
st.subheader("📈 Visualizaciones con Detalle")

tab1, tab2, tab3 = st.tabs(["Ventas vs Precio", "Curva de Optimización", "Histórico Temporal"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.regplot(data=df_p, x="price", y="Units_Sold", ax=ax1, 
                scatter_kws={'alpha':0.5, 's':60}, line_kws={'color':'red'})
    ax1.set_title("Relación Histórica: Elasticidad del Precio", fontsize=14)
    ax1.set_xlabel("Precio ($)", fontsize=12)
    ax1.set_ylabel("Unidades Vendidas (Pzs)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2b = ax2.twinx()
    
    # Graficar líneas
    l1, = ax2.plot(df_sim["Precio"], df_sim["Ingreso"], color="green", linewidth=2.5, label="Ingreso Esperado ($)")
    l2, = ax2b.plot(df_sim["Precio"], df_sim["Ventas"], color="blue", linestyle="--", label="Demanda Estimada (Pzs)")
    
    # Línea vertical de precio óptimo
    v1 = ax2.axvline(opt_row["Precio"], color="orange", linestyle=":", linewidth=2, label=f"Precio Óptimo (${opt_row['Precio']:.0f})")
    
    # Configuración de etiquetas y colores
    ax2.set_title("Simulación de Ingresos y Demanda", fontsize=14)
    ax2.set_xlabel("Rango de Precio Simulado ($)", fontsize=12)
    ax2.set_ylabel("Ingreso Total ($)", color="green", fontsize=12)
    ax2b.set_ylabel("Unidades a Vender (Pzs)", color="blue", fontsize=12)
    
    # Combinar leyendas en una sola
    labs = [l1.get_label(), l2.get_label(), v1.get_label()]
    ax2.legend([l1, l2, v1], labs, loc='upper right', frameon=True)
    
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3b = ax3.twinx()
    
    df_p["Fecha_Index"] = range(len(df_p))
    
    # Graficar datos
    p1, = ax3.plot(df_p["Fecha_Index"], df_p["Units_Sold"], color="tab:blue", marker='o', label="Unidades Vendidas")
    p2, = ax3b.plot(df_p["Fecha_Index"], df_p["price"], color="orange", linewidth=2, label="Precio Histórico")
    
    # Etiquetas
    ax3.set_title("Evolución Temporal: Comparativa de Ventas y Precio", fontsize=14)
    ax3.set_xlabel("Semanas (Historial de ventas)", fontsize=12)
    ax3.set_ylabel("Cantidad de Unidades", color="tab:blue", fontsize=12)
    ax3b.set_ylabel("Precio Unitario ($)", color="orange", fontsize=12)
    
    # Combinar leyendas
    ax3.legend([p1, p2], [p1.get_label(), p2.get_label()], loc='upper left')
    
    ax3.grid(axis='x', linestyle='--', alpha=0.5)
    st.pyplot(fig3)

if st.checkbox("Mostrar datos crudos"):
    st.write(df_p)