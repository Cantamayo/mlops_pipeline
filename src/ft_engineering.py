# librerías
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from cargar_datos import cargarDatos 

# Paso 8 (Movido arriba por buenas prácticas): Construimos la función
# Le pasamos 'X' como parámetro para que no dependa de variables globales
def ft_engineering(X):
    # Seleccionamos las columnas dinámicamente
    num_features = X.select_dtypes(include=['number']).columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns

    # Ruta 1: Numéricas
    num_transformer = Pipeline(steps=[
        # CAMBIO CLAVE: Usamos 'median' en lugar de 'mean' por los outliers extremos
        ('inputer', SimpleImputer(strategy='median'))
    ])

    # Ruta 2: Categóricas Nominales
    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('inputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combinar las 2 rutas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )
    
    return preprocessor

# Este bloque solo se ejecuta si corremos este script directamente
if __name__ == "__main__":
    print("Iniciando Pipeline de Feature Engineering...")
    
    # 1. Cargar los datos usando tu script
    df = cargarDatos()
    
    if df is not None:
        # --- APLICAMOS LAS CONCLUSIONES DEL EDA ---
        # Eliminamos las variables que dijimos que no servían o tenían 27% de nulos
        columnas_a_eliminar = [
            'fecha_prestamo', 
            'huella_consulta', 
            'tendencia_ingresos', 
            'promedio_ingresos_datacredito'
        ]
        df = df.drop(columns=columnas_a_eliminar, errors='ignore')
        print(f"\nSe eliminaron {len(columnas_a_eliminar)} columnas no deseadas.")

        # Eliminamos filas donde el Target sea nulo (no podemos predecir a ciegas)
        df = df.dropna(subset=['Pago_atiempo'])

        # Paso 1: features/target split
        X = df.drop('Pago_atiempo', axis=1) # features
        y = df['Pago_atiempo']              # target

        # Paso 5: dividir el dataset en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Obtenemos el transformador llamando a nuestra función
        preprocessor = ft_engineering(X_train)

        # Paso 6: Aplicamos el preprocesamiento
        # fit_transform aprende las reglas en Train y las aplica
        X_train_processed = preprocessor.fit_transform(X_train)
        # transform SOLO aplica las reglas en Test (para evitar Data Leakage)
        X_test_processed = preprocessor.transform(X_test)

        # Paso 7: Resultados
        print("\n¡Transformación Exitosa!")
        print(f"Dimensiones originales X_train: {X_train.shape}")
        print(f"Dimensiones de X_train_processed: {X_train_processed.shape}")
        print(f"El OneHotEncoder expandió las categorías a nuevas columnas.")