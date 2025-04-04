import json
import nltk
import unicodedata
import numpy as np
import pandas as pd
import zipfile
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from typing import List, Dict, Any
CHUNKSIZE = 2000  # Ajusta según el tamaño del dataset y el uso de memoria

# Descargar las listas de stopwords de NLTK
nltk.download('stopwords')

# Importaciones y definiciones
# Carga de movies_dataset
# Se usa en local, se reemplaza por ZIP por limitación de GitHub de 50MB
# movies_df = pd.read_csv('data/movies_dataset.csv', sep=',', encoding='utf-8', low_memory=False)
# credits_df = pd.read_csv('data/credits.csv', sep=',', encoding='utf-8', low_memory=False)

# Ruta de los archivos .zip
movies_zip_path = 'data/movies_dataset.zip'
credits_zip_path = 'data/credits.zip'

# Nos asegúramos de que ambas columnas 'id' sean del mismo tipo ver [DD]('doc/Diccionario de Datos - PIMLOps.xlsx')
#movies_df['id'] = movies_df['id'].astype(str)
#credits_df['id'] = credits_df['id'].astype(str)

movies_chunks = []
with zipfile.ZipFile(movies_zip_path, 'r') as movies_zip:
    with movies_zip.open('movies_dataset.csv') as movies_file:
        for chunk in pd.read_csv(movies_file, sep=',', encoding='utf-8', low_memory=False, chunksize=CHUNKSIZE):
            chunk['id'] = chunk['id'].astype(str)  # Convertir dentro del bucle
            movies_chunks.append(chunk)

movies_df = pd.concat(movies_chunks, ignore_index=True)

credits_chunks = []
with zipfile.ZipFile(credits_zip_path, 'r') as credits_zip:
    with credits_zip.open('credits.csv') as credits_file:
        for chunk in pd.read_csv(credits_file, sep=',', encoding='utf-8', low_memory=False, chunksize=CHUNKSIZE):
            chunk['id'] = chunk['id'].astype(str)  # Convertir dentro del bucle
            credits_chunks.append(chunk)

credits_df = pd.concat(credits_chunks, ignore_index=True)
gc.collect()

# Verificar los datos cargados
print(movies_df.head())
print(credits_df.head())
gc.collect()

columnas_a_eliminar = [ 'production_companies', 'production_countries', 'spoken_languages', 'belongs_to_collection', 'adult', 'genres', 'homepage', 'imdb_id', 'original_language', 'overview', 'poster_path', 'runtime', 'status', 'tagline', 'video' ]
movies_df = movies_df.drop(columns=columnas_a_eliminar, errors='ignore')
gc.collect()
# Verificar que se haya eliminado correctamente
print(movies_df.head())

# Definir las claves a eliminar
#cast_keys_to_remove = ['character', 'gender', 'order', 'profile_path']
#crew_keys_to_remove = ['gender', 'profile_path']

# Función para limpiar listas de diccionarios
#def limpiar_lista_diccionarios(lista, keys_to_remove):
#    if isinstance(lista, list):
#        return [{k: v for k, v in dic.items() if k not in keys_to_remove} for dic in lista]
#    return lista  # Si no es una lista, devolver el valor original

# Aplicar la función a las columnas `cast` y `crew`
#credits_df['cast'] = credits_df['cast'].apply(lambda x: limpiar_lista_diccionarios(x, cast_keys_to_remove))
#credits_df['crew'] = credits_df['crew'].apply(lambda x: limpiar_lista_diccionarios(x, crew_keys_to_remove))
gc.collect()

# Fusionar en chunks
df_chunks = []
for movies_chunk in movies_chunks:
    for credits_chunk in credits_chunks:
        merged_chunk = pd.merge(movies_chunk, credits_chunk, on='id', how='inner')
        df_chunks.append(merged_chunk)

df = pd.concat(df_chunks, ignore_index=True)  # Unimos los chunks fusionados

gc.collect()
del movies_df
del credits_df
gc.collect()

# Unir los datasets usando la columna 'id' como clave
#df = pd.merge(movies_df, credits_df, on='id', how='inner')

# Muestra las primeras filas del nuevo dataframe
print(df.head())
gc.collect()

# Transformaciones

# Los valores nulos de los campos revenue, budget deben ser rellenados por el número 0.
# Rellenar los valores nulos de las columnas 'revenue' y 'budget' con 0
df['revenue'] = df['revenue'].fillna(0)
df['budget'] = df['budget'].fillna(0)
gc.collect()
print(df.head())

# Los valores nulos del campo release date deben eliminarse.
# Eliminar las filas donde 'release_date' tenga valores nulos
df = df.dropna(subset=['release_date'])
print(df['release_date'].isnull().sum())
gc.collect()

# De haber fechas, deberán tener el formato AAAA-mm-dd, además deberán crear la columna release_year donde extraerán el año de la fecha de estreno.
#Convertir la columna 'release_date' al formato datetime:
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

#Formatear la columna 'release_date' a 'AAAA-mm-dd':
df['release_date'] = df['release_date'].dt.strftime('%Y-%m-%d')

#Crear la columna 'release_year' extrayendo el año:
df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d', errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Crear la columna con el retorno de inversión, llamada return con los campos revenue y budget, dividiendo estas dos últimas revenue / budget, cuando no hay datos disponibles para calcularlo, deberá tomar el valor 0.
# Convertir las columnas 'revenue' y 'budget' a tipo numérico, manejando los valores que no se pueden convertir
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')  # Convertir a float, poner NaN si no es posible
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')  # Convertir a float, poner NaN si no es posible

# Reemplazar valores NaN con 0 si es necesario
df['revenue'] = df['revenue'].fillna(0)
df['budget'] = df['budget'].fillna(0)

# Calcular el retorno de inversión
# Usamos np.where para manejar las divisiones por cero y los casos donde revenue es cero
df['return'] = np.where(df['budget'] != 0, df['revenue'] / df['budget'], 0)

# Asegurarse de que si tanto 'revenue' como 'budget' son 0, el retorno también sea 0
df['return'] = np.where((df['revenue'] == 0) & (df['budget'] == 0), 0, df['return'])

# Eliminar las columnas que no serán utilizadas, video,imdb_id,adult,original_title,poster_path y homepage.
# Con estas columnas consume 780MB, sobrepasa los recursos de Render.com
#columnas_a_remover = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
# Se reduce a lo minimo posible
columnas_a_remover = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage', 'genres', 'original_language', 'overview', 'production_companies', 'production_countries', 'spoken_languages', 'status', 'tagline', 'runtime', 'backdrop_path', 'gender', 'character', 'profile_path', 'order']

df = df.drop(columns=columnas_a_remover, errors='ignore')
gc.collect()

def quitar_acentos(cadena: str) -> str:
    """
    Normaliza una cadena eliminando los acentos y convirtiendo a minúsculas.
    """
    return ''.join(
        char for char in unicodedata.normalize('NFD', cadena)
        if unicodedata.category(char) != 'Mn'
    ).lower()

# Asegurarse de que la columna 'release_date' existe en el DataFrame
if 'release_date' in df.columns:
    # Convertir la columna 'release_date' a tipo datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Extraer el mes (en formato numérico)
    df['mes'] = df['release_date'].dt.month

    # Extraer el día de la semana (en formato numérico, lunes=0, domingo=6)
    df['dia_semana_num'] = df['release_date'].dt.dayofweek

    # Extraer el año
    df['anio'] = df['release_date'].dt.year

    # Para obtener el nombre del mes en español, se puede usar la configuración regional o un mapeo
    df['mes_nombre_esp'] = df['mes'].apply(lambda x: pd.to_datetime(x, format='%m').strftime('%B') if pd.notna(x) else '')

    # Para obtener el nombre del día de la semana en español, se puede usar un mapeo
    dias_semana_esp = {0: 'lunes', 1: 'martes', 2: 'miércoles', 3: 'jueves', 4: 'viernes', 5: 'sábado', 6: 'domingo'}
    df['dia_semana_nombre_esp'] = df['dia_semana_num'].map(dias_semana_esp)

    print(df[['release_date', 'mes', 'dia_semana_num', 'anio', 'mes_nombre_esp', 'dia_semana_nombre_esp']].head())
else:
    print("La columna 'release_date' no se encontró en el DataFrame.")


def obtener_cantidad_filmaciones_mes(Mes: str):
    # Mapeo de nombres de meses en español a formato de Pandas
    meses_esp_a_num = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}
    mes_num = meses_esp_a_num.get(Mes.lower())
    if mes_num:
        cantidad = len(df[df['mes'] == mes_num])
        return {"Mes": Mes, "cantidad": cantidad}
    else:
        return {"error": f"El mes '{Mes}' no es válido."}
    
def obtener_cantidad_filmaciones_dia(Dia: str):
    # Mapeo de nombres de días de la semana en español a números de Pandas
    dias_esp_a_num = {'lunes': 0, 'martes': 1, 'miercoles': 2, 'jueves': 3, 'viernes': 4, 'sabado': 5, 'domingo': 6}
    # Normalizar el argumento `Dia`
    dia_normalizado = quitar_acentos(Dia)
    dia_num = dias_esp_a_num.get(dia_normalizado)
    if dia_num is not None:
        cantidad = len(df[df['dia_semana_num'] == dia_num])
        return {"Dia": dia_normalizado, "cantidad": cantidad}
    else:
        return {"error": f"El día '{Dia}' no es válido."}

def obtener_score_titulo(titulo_de_la_filmacion: str):
    pelicula = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]
    if not pelicula.empty:
        # Considerar devolver la primera coincidencia si hay múltiples
        pelicula = pelicula.iloc[0]
        return {
            "titulo": pelicula['title'],
            "anio_estreno": int(pelicula['anio']) if pd.notna(pelicula['anio']) else None,
            "score": pelicula['popularity']
        }
    else:
        return {"mensaje": f"No se encontró la película con el título '{titulo_de_la_filmacion}'."}

def obtener_votos_titulo(titulo_de_la_filmacion: str):
    pelicula = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]
    if not pelicula.empty:
        pelicula = pelicula.iloc[0]
        cantidad_votos = pelicula['vote_count']
        if cantidad_votos >= 2000:
            return {
                "titulo": pelicula['title'],
                "cantidad_votos": int(cantidad_votos) if pd.notna(cantidad_votos) else None,
                "promedio_votos": pelicula['vote_average']
            }
        else:
            return {"mensaje": f"La película '{titulo_de_la_filmacion}' no cumple con las valoraciones mínimas (>= 2000 votos)."}
    else:
        return {"mensaje": f"No se encontró la película con el título '{titulo_de_la_filmacion}'."}

def obtener_info_actor(nombre_actor: str):
    peliculas_actor = df[df['cast'].apply(lambda x: nombre_actor.lower() in str(x).lower())]
    if not peliculas_actor.empty:
        retornos = peliculas_actor.apply(lambda row: row['revenue'] - row['budget'] if pd.notna(row['revenue']) and pd.notna(row['budget']) else np.nan, axis=1)
        retornos_validos = retornos.dropna()
        cantidad_peliculas = len(peliculas_actor)
        retorno_total = retornos_validos.sum()
        promedio_retorno = retornos_validos.mean() if not retornos_validos.empty else 0
        return {
            "nombre_actor": nombre_actor,
            "cantidad_filmaciones": cantidad_peliculas,
            "retorno_total": retorno_total,
            "promedio_retorno": promedio_retorno
        }
    else:
        return {"mensaje": f"No se encontró al actor con el nombre '{nombre_actor}' en las filmaciones."}

def extraer_directores(crew_list):
    # Verificar que 'crew_list' sea una lista válida
    if not isinstance(crew_list, list):
        return []  # Retornar lista vacía si no es lista

    # Filtrar por 'job' == 'Director' o 'department' == 'Directing'
    directores = [
        d.get('name', '').lower()
        for d in crew_list
        if isinstance(d, dict) and (d.get('job') == 'Director' or d.get('department') == 'Directing')
    ]
    return directores

def obtener_info_director(nombre_director: str):
    nombre_director_lower = nombre_director.lower()
    
    # Filtrar películas con el director especificado
    peliculas_director = df[df['crew'].apply(
        lambda x: nombre_director_lower in extraer_directores(x)
    )]
    
    if not peliculas_director.empty:
        retorno_total = peliculas_director.apply(
            lambda row: row['revenue'] - row['budget'] if pd.notna(row['revenue']) and pd.notna(row['budget']) else np.nan, axis=1
        ).dropna().sum()
        peliculas_info = []
        
        for _, pelicula in peliculas_director.iterrows():
            retorno_individual = pelicula['revenue'] - pelicula['budget'] if pd.notna(pelicula['revenue']) else None
            ganancia = pelicula['revenue'] if pd.notna(pelicula['revenue']) else None
            costo = pelicula['budget'] if pd.notna(pelicula['budget']) else None
            fecha_lanzamiento = pelicula['release_date'] if pd.notna(pelicula['release_date']) else None
            
            peliculas_info.append({
                "titulo": pelicula['title'],
                "director": nombre_director,
                "fecha_lanzamiento": fecha_lanzamiento,
                "retorno_individual": retorno_individual,
                "costo": costo,
                "ganancia": ganancia,
            })
        
        return {
            "nombre_director_buscado": nombre_director,
            "retorno_total": retorno_total,
            "peliculas": peliculas_info,
        }
    else:
        return {"mensaje": f"No se encontró al director con el nombre '{nombre_director}'."}

def obtener_recomendacion(titulo: str):
    # Obtener listas predefinidas
    stopwords_english = stopwords.words('english')
    stopwords_spanish = stopwords.words('spanish')
    stopwords_french = stopwords.words('french')
    stopwords_german = stopwords.words('german')
    stopwords_portuguese = stopwords.words('portuguese')

    # Combinar todas las listas en una sola
    combined_stopwords = stopwords_english + stopwords_spanish + stopwords_french + stopwords_german + stopwords_portuguese

    # Preparar los datos: vectorización de los títulos de las películas
    # Usar la lista combinada en TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words=combined_stopwords)
    tfidf_matrix = tfidf.fit_transform(df['title'].fillna(''))

    # Calcular similitud de coseno entre las películas
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    try:
        # Encontrar el índice de la película ingresada
        idx = df[df['title'].str.lower() == titulo.lower()].index[0]
        # Obtener similitudes de esa película con todas las demás
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Ordenar películas por puntaje de similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Obtener los índices de las 5 películas más similares
        top_indices = [i[0] for i in sim_scores[1:6]]
        # Retornar los títulos de las películas más similares
        return df.iloc[top_indices]['title'].tolist()
    except IndexError:
        return "Título no encontrado en el dataset. Por favor, verifica el nombre e intenta nuevamente."