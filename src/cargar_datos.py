 certification
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

import pandas as pd
import os

def cargar_base_datos():
    """
    Función encargada de leer el archivo Excel de la base de datos
    y retornar un DataFrame de Pandas con la información.
    """
    # 1. Obtenemos la ruta absoluta de la carpeta donde está este script (src)
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construimos la ruta uniendo esa ubicación con '../Base_de_datos.xlsx'
    ruta_archivo = os.path.join(directorio_actual, '..', 'Base_de_datos.xlsx')
    
    print(f"Iniciando proceso de carga...")
    print(f"Buscando el archivo en la ruta exacta:\n{ruta_archivo}")
    
    try:
        # Intentamos leer el archivo Excel
        df = pd.read_excel(ruta_archivo)
        
        # Si funciona, mostramos un mensaje de éxito y el tamaño de los datos
        print("\n¡Carga exitosa!")
        print(f"El dataset tiene {df.shape[0]} filas (clientes) y {df.shape[1]} columnas (variables).")
        
        # Mostramos los nombres de las columnas para verificar qué datos tenemos
        print("\nColumnas encontradas:")
        print(df.columns.tolist())
        
        return df
        
    except FileNotFoundError:
        print(f"\n[ERROR] No se encontró el archivo en la ruta mostrada arriba.")
        print("Asegúrate de que el archivo se llame exactamente 'Base_de_datos.xlsx'")
        return None
    except ImportError:
        print(f"\n[ERROR] Falta una dependencia para leer archivos de Excel.")
        print("Por favor instala 'openpyxl' escribiendo en tu terminal: pip install openpyxl")
        return None
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un problema inesperado: {e}")
        return None


if __name__ == "__main__":
    datos = cargar_base_datos()
 main
