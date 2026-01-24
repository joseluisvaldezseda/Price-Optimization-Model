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

# Page Configuration
st.set_page_config(page_title="Pricing Optimization Tool", layout="wide")

# --- DATA LOADING AND PROCESSING FUNCTIONS ---
@st.cache_data
def load_and_clean_data():
    # Note: Ensure the file exists in the path
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
    
    # Calculate basic unit price for correlation calculations
    df["price"] = df["Total_Sales"] / df["Units_Sold"].replace(0, np.nan)
    return df

@st.cache_data
def get_product_stats(df):
    """Calcula semanas reales (únicas) y correlación sobre datos agrupados"""
    stats = []
    # Agrupamos primero por ID, Año y Semana para tener valores únicos por semana
    # Esto asegura que el conteo sea igual al que verás en el análisis
    df_weekly = df.groupby(["Product_ID", "Year", "Week"]).agg({
        "Total_Sales": "sum",
        "Units_Sold": "sum",
        "Product_Description": "first"
    }).reset_index()
    
    # Calcular precio por semana
    df_weekly["price"] = df_weekly["Total_Sales"] / df_weekly["Units_Sold"].replace(0, np.nan)

    for pid, group in df_weekly.groupby("Product_ID"):
        # Limpiar nans para la correlación
        valid_data = group.dropna(subset=['price', 'Units_Sold'])
        
        # Ahora len(group) es el número de SEMANAS reales
        n_weeks = len(group)
        
        if n_weeks < 2: continue
        
        corr = valid_data["price"].corr(valid_data["Units_Sold"])
        
        stats.append({
            "Product_ID": pid,
            "Weeks": n_weeks, # Semanas reales agrupadas
            "Corr": abs(corr) if not np.isnan(corr) else 0,
            "Description": group["Product_Description"].iloc[0]
        })
    return pd.DataFrame(stats)


@st.cache_data
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

@st.cache_resource
def train_best_model(df_p):
    X = df_p[["price", "week_sin", "week_cos"]]
    y = df_p["Units_Sold"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    best_r2, best_model, best_name = -np.inf, None, ""
    for name, model in models.items():
        model.fit(X_scaled, y)
        r2 = r2_score(y, model.predict(X_scaled))
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name
    
    return best_model, scaler, best_name, best_r2

# --- USER INTERFACE ---
st.title("Price Optimizer")

try:
    df_raw = load_and_clean_data()
    # Generate stats table for quick filtering
    df_stats = get_product_stats(df_raw)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Analysis Filters")
min_rows = st.sidebar.slider("Minimum weeks of history", 5, 150, 100)
seuil_corr = st.sidebar.slider("Correlation Threshold (Elasticity)", 0.0, 1.0, 0.2)

# Apply BOTH filters to the product list
mask = (df_stats["Weeks"] >= min_rows) & (df_stats["Corr"] >= seuil_corr)
df_filtered_list = df_stats[mask]

valid_products = df_filtered_list["Product_ID"].tolist()

if not valid_products:
    st.warning(f"No products meet both criteria (Weeks >= {min_rows} and Correlation >= {seuil_corr}).")
    st.stop()

# Show how many products passed the filter
st.sidebar.info(f"Products found: {len(valid_products)}")

# Product selector based on filtered list
selected_id = st.selectbox("Select a Product (Product_ID)", valid_products)

# Get selected product data
product_info = df_raw[df_raw["Product_ID"] == selected_id].iloc[0]
st.subheader(f"{product_info['Product_Description']} | Brand: {product_info['Brand']}")

df_p = process_product(df_raw[df_raw["Product_ID"] == selected_id])

# Quick metrics
corr = df_p["price"].corr(df_p["Units_Sold"])
col1, col2, col3, col4 = st.columns(4)
col1.metric("Data Weeks", len(df_p))
col2.metric("Elasticity (Corr)", f"{corr:.2f}")
col3.metric("Average Price", f"${df_p['price'].mean():,.2f}")
col4.metric("Total Sales (Qty)", f"{df_p['Units_Sold'].sum():,.0f}")

st.divider()

# --- MODELING LOGIC AND VISUALIZATION ---
with st.spinner('Analyzing demand patterns...'):
    best_model, scaler, best_name, best_r2 = train_best_model(df_p)
    st.write(f"Best model: **{best_name}** (R²: {best_r2:.3f})")

    prix_range = np.linspace(df_p["price"].min() * 0.8, df_p["price"].max() * 1.2, 50)
    next_week = (df_p["Week"].max() + 1) % 52
    
    df_sim_X = pd.DataFrame({
        "price": prix_range,
        "week_sin": np.sin(2 * np.pi * next_week / 52),
        "week_cos": np.cos(2 * np.pi * next_week / 52)
    })
    
    X_opt_scaled = scaler.transform(df_sim_X)
    preds = best_model.predict(X_opt_scaled)
    preds = np.maximum(0, preds)
    
    df_sim = pd.DataFrame({
        "Price": prix_range, "Sales": preds, "Revenue": prix_range * preds
    })
    
    opt_row = df_sim.loc[df_sim["Revenue"].idxmax()]

col_a, col_b = st.columns(2)
with col_a:
    st.success(f"**Optimal Price: ${opt_row['Price']:,.2f}**")
with col_b:
    st.info(f"**Estimated Revenue: ${opt_row['Revenue']:,.2f}**")

# --- VISUALIZATIONS ---
st.divider()
st.subheader("📈 Detailed Visualizations")

tab1, tab2, tab3 = st.tabs(["Sales vs Price", "Optimization Curve", "Temporal History"])

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "grid.color": "#EEEEEE",
    "font.family": "sans-serif",
    "legend.frameon": True,
    "legend.fontsize": 10
})

with tab1:
    #fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    
    sns.regplot(
        data=df_p, x="price", y="Units_Sold", ax=ax1, 
        scatter_kws={'alpha':0.4, 's':80, 'color':'#2c3e50', 'edgecolor':'white'}, 
        line_kws={'color':'#e74c3c', 'linewidth': 3, 'label': 'Demand Trend'}
    )
    
    ax1.scatter(
        opt_row["Price"], opt_row["Sales"], 
        color="#f1c40f", s=400, marker="*", 
        edgecolor="#2c3e50", linewidth=1.5, label="OPTIMAL RECOMMENDATION", zorder=10
    )
    
    ax1.set_title("Historical Relationship: Price vs Demand", fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax1.set_xlabel("Selling Price ($)", fontsize=13, labelpad=10)
    ax1.set_ylabel("Units Sold (Qty)", fontsize=13, labelpad=10)
    
    ax1.legend(facecolor='white', edgecolor='#dddddd')
    ax1.grid(True, linestyle='--', alpha=0.7)
    sns.despine(left=True, bottom=True)
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2b = ax2.twinx()
    
    ax2.fill_between(df_sim["Price"], df_sim["Revenue"], color="#2ecc71", alpha=0.2)
    l1, = ax2.plot(df_sim["Price"], df_sim["Revenue"], color="#27ae60", linewidth=3, label="Expected Revenue ($)")
    
    l2, = ax2b.plot(df_sim["Price"], df_sim["Sales"], color="#2980b9", linestyle="--", linewidth=2, label="Estimated Demand (Units)")
    
    v1 = ax2.axvline(opt_row["Price"], color="#e67e22", linestyle=":", linewidth=2.5, label=f"Optimal Price: ${opt_row['Price']:.0f}")
    
    ax2.set_title("Revenue Optimization Simulation", fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax2.set_xlabel("Simulated Price Range ($)", fontsize=13)
    ax2.set_ylabel("Total Revenue ($)", color="#27ae60", fontsize=13, fontweight='bold')
    ax2b.set_ylabel("Units to Sell (Qty)", color="#2980b9", fontsize=13, fontweight='bold')
    
    labs = [l1.get_label(), l2.get_label(), v1.get_label()]
    ax2.legend([l1, l2, v1], labs, loc='upper right', frameon=True, shadow=True)
    
    ax2.grid(True, axis='y', alpha=0.3)
    sns.despine(ax=ax2, right=False, left=False)
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    ax3b = ax3.twinx()
    
    df_p["Fecha_Index"] = range(len(df_p))
    idx_proyeccion = len(df_p)
    
    ax3.fill_between(df_p["Fecha_Index"], df_p["Units_Sold"], color="#3498db", alpha=0.1)
    
    p1, = ax3.plot(df_p["Fecha_Index"], df_p["Units_Sold"], color="#3498db", marker='o', markersize=4, label="Units Sold", linewidth=2)
    p2, = ax3b.plot(df_p["Fecha_Index"], df_p["price"], color="#f39c12", linewidth=2, alpha=0.8, label="Historical Price")
    
    p3 = ax3.scatter(idx_proyeccion, opt_row["Sales"], color="#2980b9", marker='D', s=150, label="Sales Projection", zorder=10, edgecolor='white')
    p4 = ax3b.scatter(idx_proyeccion, opt_row["Price"], color="#c0392b", marker='D', s=150, label="Suggested Price", zorder=10, edgecolor='white')
    
    ax3.axvline(x=len(df_p)-0.5, color="#cccccc", linestyle="--", linewidth=1)
    ax3.text(len(df_p)-0.5, ax3.get_ylim()[1], ' PROJECTION', color='#999999', fontsize=9, va='bottom')

    ax3.plot([len(df_p)-1, idx_proyeccion], [df_p["Units_Sold"].iloc[-1], opt_row["Sales"]], color="#3498db", linestyle="--", alpha=0.5)
    ax3b.plot([len(df_p)-1, idx_proyeccion], [df_p["price"].iloc[-1], opt_row["Price"]], color="#f39c12", linestyle="--", alpha=0.5)

    ax3.set_title("Temporal Evolution and Price Suggestion", fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
    ax3.set_xlabel("Weeks (History + Next Week)", fontsize=13, labelpad=15)
    ax3.set_ylabel("Unit Quantity", color="#3498db", fontsize=13, fontweight='bold')
    ax3b.set_ylabel("Unit Price ($)", color="#f39c12", fontsize=13, fontweight='bold')
    
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3b.get_legend_handles_labels()
    
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), 
               ncol=4, 
               frameon=True, 
               shadow=False, 
               fontsize=11)
    
    ax3.grid(axis='x', linestyle='-', alpha=0.1)
    sns.despine(ax=ax3, right=False)
    
    plt.tight_layout()
    st.pyplot(fig3)

if st.checkbox("Show raw data"):

    st.write(df_p)
