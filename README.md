# Proyecto de Análisis de Películas y API con FastAPI

Este proyecto implementa un sistema de análisis y consulta de datos de películas mediante # Proyecto de Análisis de Películas y API con FastAPIla creación de un **API
RESTful** utilizando **FastAPI**. Integra procesamiento de datos en Python con `pandas`,análisis de texto con `TfidfVectorizer` y medidas de similitud como **cosine similarity**. Además, se incluye un preprocesamiento detallado de los datos para extraer información relevante.

## Descripción General

El objetivo del proyecto es procesar un conjunto de datos de películas (`movies_dataset.csv` y `credits.csv`), enriquecerlo con transformaciones útiles y crear varios endpoints que permitan a los usuarios realizar consultas como:
- Cantidad de filmaciones en un mes o día específico.
- Información de películas por título, director o actor.
- Recomendación de películas similares basadas en un sistema de puntuación de similitud.

---

## Funcionalidades Principales

### 1. **Procesamiento de Datos**
- Unificación de los datasets `movies_dataset.csv` y `credits.csv` a través de la columna `id`.
- Limpieza de datos:
  - Valores nulos en `revenue`, `budget` son reemplazados por `0`.
  - Eliminación de filas con valores nulos en `release_date`.
- Transformaciones:
  - Conversión de las fechas al formato `AAAA-mm-dd`.
  - Extracción de columnas como `release_year` y `mes`.
  - Cálculo del retorno de inversión (`return = revenue / budget`).
- Desanidación de campos JSON como `genres`, `spoken_languages`, entre otros.

### 2. **Creación de la API REST**
Se implementaron los siguientes endpoints:

- **`/cantidad_filmaciones_mes/{Mes}`**: Retorna la cantidad de filmaciones en un mes específico (ingresado en español).
- **`/cantidad_filmaciones_dia/{Dia}`**: Retorna la cantidad de filmaciones en un día específico (ingresado en español).
- **`/score_titulo/{titulo_de_la_filmacion}`**: Devuelve el puntaje de popularidad y el año de estreno de una película por su título.
- **`/votos_titulo/{titulo_de_la_filmacion}`**: Proporciona el número de votos y el promedio de votos de una película, siempre que tenga más de 2000 votos.
- **`/get_actor/{nombre_actor}`**: Informa el total de filmaciones, el retorno promedio y el total de ingresos para un actor.
- **`/get_director/{nombre_director}`**: Proporciona una lista de películas dirigidas por un director, junto con información financiera (costo, ganancia, retorno).
- **`/recomendacion/{titulo}`**: Recomienda 5 películas similares basadas en el título proporcionado, utilizando similitud de coseno en títulos.

---

## Requisitos del Sistema

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias de Python:
```bash
pip install pandas numpy fastapi scikit-learn uvicorn nltk
```

Además, debes descargar las *stopwords* de **NLTK**:
```python
import nltk
nltk.download('stopwords')
```

---

## Estructura del Proyecto

```
proyecto/
├── app/
│   ├── api/
│   │    └── endpoints.py
│   ├── img/
│   │    └── favicon.ico
│   ├── transformations/
│   │    └── data_processing.py
│   └── main.py
├── data/
│   ├── credits.zip
│   └── movies_dataset.zip
├── doc/
│   └── Diccionario de Datos - PIMLOps.xlsx
├── README.md
└── tp1.ipynb
```
- **app/**: Contiene los modulos de la app.
- **app/api/**: Contiene los enpoints.
- **app/img/**: Contiene icono de la web.
- **app/transformations/**: Contiene las transformaciones aplicadas.
- **data/**: Contiene los datasets necesarios.
- **doc/**: Incluye documentación relevante como el diccionario de datos.
- **main.py**: Código principal del proyecto y definición de los endpoints.
- **README.md**: Documentación del proyecto.

---

## Cómo Ejecutar el Proyecto

1. **Preparación de los datos**:
   - Asegúrate de que los archivos `movies_dataset.csv` y `credits.csv` estén en la carpeta `DATA/`.

2. **Iniciar la API**:
   Ejecuta el archivo `main.py` con Uvicorn:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   
   ## o bien
   
   & "/.../python.exe" /.../main.py
   ```
   Esto iniciará un servidor local en `http://127.0.0.1:8000`.

3. **Probar los Endpoints**:
   Abre tu navegador o usa herramientas como **Postman** o **cURL** para acceder a los endpoints, por ejemplo:
   - `http://127.0.0.1:8000/cantidad_filmaciones_mes/enero`
   - `http://127.0.0.1:8000/recomendacion/Titanic`

---

## Detalles Técnicos

- **Procesamiento del Lenguaje Natural**:
  Se utilizan las listas de *stopwords* de **NLTK** en múltiples idiomas (inglés, español, francés, alemán, portugués) para mejorar la precisión del modelo de recomendación.

- **Modelo de Recomendación**:
  Utiliza `TfidfVectorizer` para convertir los títulos de las películas en vectores, y luego calcula la similitud de coseno entre ellos para determinar las películas más similares.

- **Preprocesamiento de Datos**:
  Incluye la eliminación de columnas no utilizadas (`video`, `imdb_id`, `adult`, etc.) para optimizar el tamaño del dataset.

---

## Próximos Pasos

1. **Ampliar Soporte Multilenguaje**:
   Extender el uso de *stopwords* a más idiomas según la lista de idiomas únicos en el dataset.

2. **Mejorar el Sistema de Recomendación**:
   Incorporar información adicional como géneros o actores para aumentar la precisión de las recomendaciones.

3. **Documentación**:
   Añadir ejemplos específicos para cada endpoint en la API.

---

## Contribuciones

Si deseas colaborar o reportar errores, puedes crear un issue en este repositorio o enviar tus propuestas de mejora. ¡Toda ayuda es bienvenida!

---

## Licencia

Este proyecto se publica bajo la Licencia MIT. Para más información, consulta el archivo `LICENSE`.

```
