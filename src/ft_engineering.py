import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from cargar_datos import cargarDatos 

# --- 1. REGLAS DE NEGOCIO Y CALIDAD DE DATOS ---
def aplicar_filtros_eda(df):
    df_clean = df.copy()
    
    # Filtros de Data Quality exigidos
    df_clean = df_clean[df_clean['salario_cliente'] > 0]
    df_clean = df_clean[(df_clean['edad_cliente'] >= 18) & (df_clean['edad_cliente'] <= 90)]
    df_clean = df_clean[df_clean['cuota_pactada'] <= df_clean['capital_prestado']]
    
    # Manejo de negativos (Sin Historial -> NaN)
    if 'puntaje' in df_clean.columns:
        df_clean.loc[df_clean['puntaje'] < 0, 'puntaje'] = np.nan
    if 'puntaje_datacredito' in df_clean.columns:
        df_clean.loc[df_clean['puntaje_datacredito'] < 0, 'puntaje_datacredito'] = np.nan
        
    return df_clean

# --- 2. ATRIBUTOS DERIVADOS ---
def crear_variables_nuevas(df):
    df_der = df.copy()
    # carga_financiera
    df_der['carga_financiera'] = df_der['cuota_pactada'] / df_der['salario_cliente']
    
    # saldo_restante (solo si saldo_principal no ha sido eliminada por Data Leakage)
    if 'saldo_principal' in df_der.columns and 'capital_prestado' in df_der.columns:
        df_der['saldo_restante'] = df_der['capital_prestado'] - df_der['saldo_principal']
        
    return df_der

# --- 3. PIPELINE DE TRANSFORMACIÓN ---
def convertir_a_string(x):
    return x.astype(str)

def ft_engineering(X):
    # Variables sesgadas que requieren np.log1p según EDA
    # (Añadimos total_otros_prestamos por su altísima varianza en la tabla)
    sesgadas_cols = ['salario_cliente', 'capital_prestado', 'total_otros_prestamos']
    sesgadas_features = [col for col in sesgadas_cols if col in X.columns]
    
    # Categóricas
    cat_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Numéricas estándar (las que sobran)
    num_features = [col for col in X.select_dtypes(include=['number']).columns if col not in sesgadas_features]

    # Ruta 1: Numéricas con asimetría (Log transform)
    log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p))
    ])

    # Ruta 2: Numéricas normales
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Ruta 3: Categóricas (OneHotEncoder exigido en el EDA)
    cat_transformer = Pipeline(steps=[
    ('to_str', FunctionTransformer(convertir_a_string)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

    preprocessor = ColumnTransformer(
        transformers=[
            ('log_num', log_transformer, sesgadas_features),
            ('std_num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ], remainder='passthrough'
    )
    
    return preprocessor

if __name__ == "__main__":
    print("Iniciando Pipeline de Feature Engineering...")
    df = cargarDatos()
    
    if df is not None:
        # Aplicar Reglas de Calidad y Derivadas antes de modelar
        df = aplicar_filtros_eda(df)
        df = crear_variables_nuevas(df)
        
        columnas_a_eliminar = [
            'fecha_prestamo', 'huella_consulta', 'tendencia_ingresos', 
            'promedio_ingresos_datacredito', 'puntaje', 'puntaje_datacredito',
            'saldo_mora', 'saldo_mora_codeudor', 'saldo_total', 'saldo_principal'
        ]
        df = df.drop(columns=columnas_a_eliminar, errors='ignore').dropna(subset=['Pago_atiempo'])

        X = df.drop('Pago_atiempo', axis=1)
        y = df['Pago_atiempo']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        preprocessor = ft_engineering(X_train)

        X_train_processed = preprocessor.fit_transform(X_train)
        print(f"\n¡Transformación Exitosa! Nuevas dimensiones: {X_train_processed.shape}")