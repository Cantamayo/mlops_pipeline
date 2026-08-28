# src/model_monitoring.py

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.model_selection import train_test_split

# Importamos tu función de carga de datos (usando el mismo nombre del Avance 2)
from cargar_datos import cargar_base_datos

######################################################
####### 1. Configuración de la aplicación ############
######################################################
st.set_page_config(page_title="MLOps - Riesgo Crediticio", layout="wide")

#######################################################
###### 2. Cargar el dataset y dividir los datos ######
#######################################################
@st.cache_data
def load_dataset():
    # 2.1 Cargamos los datos originales
    df = cargar_base_datos()
    
    # Aplicamos una limpieza rápida 
    df = df.dropna(subset=['Pago_atiempo'])
    
    # 2.2 ¡CORRECCIÓN CLAVE! La 'P' es mayúscula y lo convertimos a 1 (Moroso)
    target = "Pago_atiempo"
    y = (df[target] == 0).astype(int) 
    
    # Eliminamos columnas que sabemos que son Data Leakage
    cols_drop = [target, 'fecha_prestamo', 'huella_consulta', 'tendencia_ingresos', 
                 'promedio_ingresos_datacredito', 'saldo_mora', 'saldo_mora_codeudor', 
                 'saldo_total', 'saldo_principal', 'puntaje', 'puntaje_datacredito']
    X = df.drop(columns=[c for c in cols_drop if c in df.columns])

    # 2.3 Dividimos los datos (Simulando: X_ref es el pasado, X_new es el presente)
    X_ref, X_new, y_ref, y_new = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_ref, X_new

X_ref, X_new = load_dataset()

#######################################################
###### 3. Lógica Matemática de Data Drift (Rúbrica) ###
#######################################################
def calcular_drift(ref, new):
    resultados = []
    
    # Separar numéricas y categóricas
    num_cols = ref.select_dtypes(include=['number']).columns
    cat_cols = ref.select_dtypes(include=['object', 'category']).columns
    
    # 3.1 Kolmogorov-Smirnov (KS test) para Numéricas
    for col in num_cols:
        stat, p_value = ks_2samp(ref[col].dropna(), new[col].dropna())
        drift = "🔴 Sí" if p_value < 0.05 else "🟢 No"
        resultados.append({"Variable": col, "Tipo": "Numérica", "Test": "KS Test", 
                           "P-Value": round(p_value, 4), "Data Drift": drift})
        
    # 3.2 Chi-cuadrado para Categóricas
    for col in cat_cols:
        ref_counts = ref[col].value_counts()
        new_counts = new[col].value_counts()
        df_counts = pd.DataFrame({'ref': ref_counts, 'new': new_counts}).fillna(0)
        
        if df_counts.shape[0] > 1:
            stat, p_value, dof, expected = chi2_contingency(df_counts)
            drift = "🔴 Sí" if p_value < 0.05 else "🟢 No"
        else:
            p_value = 1.0
            drift = "🟢 No"
            
        resultados.append({"Variable": col, "Tipo": "Categórica", "Test": "Chi-Cuadrado", 
                           "P-Value": round(p_value, 4), "Data Drift": drift})
        
    return pd.DataFrame(resultados)

##########################################
##### 4. Interfaz de Streamlit (UI) ######
##########################################

st.title("🚀 Dashboard MLOps - Monitoreo de Data Drift")
st.markdown("Sistema de detección de cambios distribucionales en la población de riesgo crediticio.")

# --- SECCIÓN A: TABLA DE MÉTRICAS ---
st.header("1. Análisis Estadístico de Variables")
df_drift = calcular_drift(X_ref, X_new)

variables_con_drift = len(df_drift[df_drift["Data Drift"] == "🔴 Sí"])

# --- SEMÁFOROS Y ALERTAS (Rúbrica) ---
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Variables Analizadas", value=len(df_drift))
with col2:
    if variables_con_drift > 0:
        st.metric(label="Estado de Salud del Modelo", value="CRÍTICO 🔴", delta=f"{variables_con_drift} variables desviadas", delta_color="inverse")
        st.error("⚠️ **Alerta:** Se ha detectado Data Drift. Se recomienda pausar predicciones automatizadas e iniciar ciclo de Retraining.")
    else:
        st.metric(label="Estado de Salud del Modelo", value="ESTABLE 🟢", delta="Sin desviaciones", delta_color="normal")
        st.success("✅ **Modelo Estable:** Las distribuciones actuales coinciden con la data de entrenamiento.")

st.dataframe(df_drift, use_container_width=True)

# --- SECCIÓN B: VISUALIZACIÓN DE MÉTRICAS (Rúbrica) ---
st.header("2. Comparación Histórica vs. Actual")
st.markdown("Visualización de las distribuciones para validar gráficamente el Drift.")

variable_seleccionada = st.selectbox("Selecciona una variable numérica para graficar:", X_ref.select_dtypes(include=['number']).columns)

if variable_seleccionada:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(X_ref[variable_seleccionada].dropna(), fill=True, label="Histórico (Train)", ax=ax, color="blue", alpha=0.3)
    sns.kdeplot(X_new[variable_seleccionada].dropna(), fill=True, label="Actual (Producción)", ax=ax, color="orange", alpha=0.3)
    plt.title(f"Distribución de {variable_seleccionada}")
    plt.legend()
    st.pyplot(fig)