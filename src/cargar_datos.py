import os
import pandas as pd
def cargarDatos():

    #1. Ruta absoluta del directorio donde esta este archivo (src)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    #2. Subir un nivel para llegar a la carpeta donde esta la base de datos
    ruta_proyecto = os.path.dirname(ruta_actual)

    #3. Construir la ruta completa al archivo Excel
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")

    #4. leemos los datos y los imprimimos
    df = pd.read_excel(ruta_excel)
    print(df)
    return df

if __name__ == "__main__":
    # Si se ejecuta este script directamente, carga los datos y muestra las primera filas
    datos = cargarDatos()
    print(datos.head())
    print(datos.columns)