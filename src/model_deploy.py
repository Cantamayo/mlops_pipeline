# src/model_deploy.py

import pandas as pd
import joblib
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

###############################
### 1. Inicialización de la API
###############################

app = FastAPI(
    title="API de Riesgo Crediticio (MLOps)",
    description="Motor predictivo para detectar clientes Morosos (Alto Riesgo) basado en un pipeline de LightGBM.",
    version="2.0.0"
)

###################################
### 2. Cargamos el modelo entrenado
###################################
modelo = None

try:
    # Aseguramos la ruta correcta donde guardaste el .pkl en el Avance 2
    # os.path.dirname(__file__) nos ubica en la carpeta 'src' dinámicamente
    model_path = os.path.join(os.path.dirname(__file__), "modelo_riesgo_crediticio.pkl")
    modelo = joblib.load(model_path)
    print("✅ Pipeline y Modelo LightGBM cargados correctamente.")

except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

########################################
### 3. Definimos los endpoints de la API
########################################

# 3.1 Endpoint de saludo (Health Check)
@app.get("/saludo")
def saludo():
    return {
        "mensaje": "¡Hola! La API está activa.",
        "estado": "Operativo",
        "objetivo": "Detección de impagos (Morosos = 1, Buenos Pagadores = 0)"
    }

# 3.2 Endpoint de predicción (Soporta Batch)
@app.post("/predict")
def predict_batch(input_data: List[Dict[str, Any]]):
    """
    Recibe una lista de diccionarios (JSON) con los datos de múltiples clientes.
    """
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no pudo ser cargado en el servidor.")

    try:
        # 1. Convertir el JSON entrante a un DataFrame de Pandas
        df_nuevos_datos = pd.DataFrame(input_data)
        
        # 2. El Pipeline hace toda la limpieza y genera las predicciones
        predicciones = modelo.predict(df_nuevos_datos)
        probabilidades = modelo.predict_proba(df_nuevos_datos)[:, 1] # Probabilidad de ser Clase 1 (Moroso)
        
        # 3. Formatear la respuesta para el usuario final
        resultados = []
        for pred, prob in zip(predicciones, probabilidades):
            resultados.append({
                "prediccion": int(pred),
                "probabilidad_riesgo": round(float(prob), 4),
                "etiqueta": "🔴 Alto Riesgo (Moroso)" if pred == 1 else "🟢 Bajo Riesgo (A tiempo)"
            })
            
        return {"total_procesados": len(resultados), "predicciones": resultados}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar los datos: {str(e)}")

# Bloque para correr localmente sin usar uvicorn desde consola (Opcional)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)