import os

def actualizar_diccionario(carpetas_origen, archivo_salida="RouterKeygen.dic"):
    claves_unicas = set()
    archivos_procesados = 0

    print("=== Iniciando actualización de RouterKeygen.dic ===")
    
    # Buscar en todos los archivos de texto de las carpetas indicadas
    for carpeta in carpetas_origen:
        if not os.path.exists(carpeta):
            print(f"Advertencia: La carpeta '{carpeta}' no existe. Omitiendo...")
            continue
            
        for archivo in os.listdir(carpeta):
            if archivo.endswith(".txt") or archivo.endswith(".lst"):
                ruta_completa = os.path.join(carpeta, archivo)
                print(f"Procesando: {ruta_completa}")
                archivos_procesados += 1
                
                # Leer contraseñas eliminando espacios y saltos de línea
                with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                    for linea in f:
                        clave = linea.strip()
                        # Filtrar cadenas vacías o extremadamente cortas
                        if len(clave) >= 8: 
                            claves_unicas.add(clave)

    # Escribir las claves ordenadas en el nuevo archivo .dic
    print(f"\nEscribiendo {len(claves_unicas)} claves únicas en {archivo_salida}...")
    with open(archivo_salida, "w", encoding="utf-8") as f_salida:
        for clave in sorted(claves_unicas):
            f_salida.write(f"{clave}\n")
            
    print(f"¡Proceso completado! Se consolidaron {archivos_procesados} archivos.")

# Configuración de ejecución
if __name__ == "__main__":
    # Coloca aquí los nombres de las carpetas donde guardas tus diccionarios nuevos (.txt)
    carpetas_diccionarios = ["./nuevos_diccionarios", "./listas_wpa"]
    
    # Crear las carpetas de ejemplo si no existen
    for c in carpetas_diccionarios:
        os.makedirs(c, exist_ok=True)
        
    actualizar_diccionario(carpetas_diccionarios)
              
