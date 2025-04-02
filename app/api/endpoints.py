import nltk
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from typing import List
from app.transformations.data_processing import *

router = APIRouter()

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/img/favicon.ico")

@router.get("/cantidad_filmaciones_mes/{Mes}")
async def cantidad_filmaciones_mes(Mes: str):
    """
    Obtiene cantidad de filmaciones por mes.
    """
    return obtener_cantidad_filmaciones_mes(Mes)

# %% [markdown]
# Implementación del Endpoint /cantidad_filmaciones_dia/{Dia}

# %%
@router.get("/cantidad_filmaciones_dia/{Dia}")
async def cantidad_filmaciones_dia(Dia: str):
    """
    Obtiene cantidad de filmaciones por dia.
    """
    return obtener_cantidad_filmaciones_dia(Dia)
    
# %% [markdown]
# Implementación del Endpoint /score_titulo/{titulo_de_la_filmacion}

# %%
@router.get("/score_titulo/{titulo_de_la_filmacion}")
async def score_titulo(titulo_de_la_filmacion: str):
    """
    Obtiene el titulo, año y puntuacion de una pelicula.
    """
    return obtener_score_titulo(titulo_de_la_filmacion) 
    
# %% [markdown]
# Implementación del Endpoint /votos_titulo/{titulo_de_la_filmacion}

# %%
@router.get("/votos_titulo/{titulo_de_la_filmacion}")
async def votos_titulo(titulo_de_la_filmacion: str):
    """
    Obtiene el titulo, cantidad de votos y promedio de una pelicula.
    """
    return obtener_votos_titulo(titulo_de_la_filmacion)

# %% [markdown]
# Implementación del Endpoint /get_actor/{nombre_actor}

# %%
@router.get("/get_actor/{nombre_actor}")
async def get_actor(nombre_actor: str):
    """
    Obtiene el nombre del actor, cantidad de filmaciones, retorno total y promedio de retorno de un actor.
    """
    return obtener_info_actor(nombre_actor)
    
# %% [markdown]
# Implementación del Endpoint /get_director/{nombre_director}

# %%
@router.get("/get_director/{nombre_director}")
async def get_director(nombre_director: str):
    """
    Obtiene el nombre del actor, cantidad de filmaciones, retorno total y promedio de retorno de un actor.
    """
    return obtener_info_director(nombre_director)
     
# %%
@router.get("/recomendacion/{titulo}")
async def recomendacion(titulo: str):
    """
    Obtiene recomendaciones de películas similares basado en el título ingresado.
    """
    return obtener_recomendacion(titulo)