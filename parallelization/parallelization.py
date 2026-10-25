r"""
Workflow: Parallelization con LangGraph + prompts de LangSmith.

Varias llamadas al LLM trabajan AL MISMO TIEMPO sobre el mismo input, cada una
en una subtarea independiente. Un agregador (código, sin LLM) junta los
resultados cuando todas terminan.
Cada llamada es un prompt guardado en LangSmith (Prompt Hub); el modelo y su
proveedor se configuran allí. Aquí solo usamos el identificador del prompt.

Ejemplo: analizar la reseña de un cliente.

Flujo:
              /-> sentimiento -\
    START ----+-> resumen     --+--> agregador -> END
              \-> temas       -/

Ejecutar:
    python parallelization.py "El envío tardó 10 días, pero el producto es excelente"
"""

import os
import sys
from functools import lru_cache
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langsmith import Client

load_dotenv()

# Identificadores de los prompts en LangSmith (Prompt Hub).
# Formato: "nombre" o "nombre:commit" para fijar una versión concreta.
PROMPT_SENTIMIENTO = os.getenv("PROMPT_SENTIMIENTO", "par-sentimiento")  # variables: {input}
PROMPT_RESUMEN = os.getenv("PROMPT_RESUMEN", "par-resumen")              # variables: {input}
PROMPT_TEMAS = os.getenv("PROMPT_TEMAS", "par-temas")                    # variables: {input}


@lru_cache
def cargar_prompt(prompt_id: str):
    """Descarga el prompt desde LangSmith junto con su modelo configurado.

    include_model=True devuelve un Runnable (prompt | modelo), listo para .invoke().
    Se cachea para no volver a descargarlo en cada llamada.
    """
    return Client().pull_prompt(prompt_id, include_model=True)


def ejecutar_prompt(prompt_id: str, variables: dict) -> str:
    """Invoca el prompt y devuelve el texto de la respuesta del modelo."""
    respuesta = cargar_prompt(prompt_id).invoke(variables)
    return respuesta.content if hasattr(respuesta, "content") else str(respuesta)


# --- Estado -----------------------------------------------------------------
# Cada rama escribe en SU PROPIA clave, así las ramas paralelas no se pisan.
class AgentState(TypedDict, total=False):
    input: str        # reseña del cliente
    sentimiento: str  # salida de la rama 1
    resumen: str      # salida de la rama 2
    temas: str        # salida de la rama 3
    reporte: str      # salida del agregador


# --- Ramas paralelas --------------------------------------------------------
def analizar_sentimiento(state: AgentState) -> AgentState:
    # Entra: input  ->  Sale: sentimiento
    return {"sentimiento": ejecutar_prompt(PROMPT_SENTIMIENTO, {"input": state["input"]})}


def resumir(state: AgentState) -> AgentState:
    # Entra: input  ->  Sale: resumen
    return {"resumen": ejecutar_prompt(PROMPT_RESUMEN, {"input": state["input"]})}


def extraer_temas(state: AgentState) -> AgentState:
    # Entra: input  ->  Sale: temas
    return {"temas": ejecutar_prompt(PROMPT_TEMAS, {"input": state["input"]})}


# --- Agregador --------------------------------------------------------------
def agregador(state: AgentState) -> AgentState:
    """Junta los resultados de las 3 ramas. Es código normal, sin LLM.

    LangGraph lo ejecuta UNA sola vez, cuando las 3 ramas han terminado.
    Entra: sentimiento, resumen, temas  ->  Sale: reporte
    """
    reporte = (
        f"SENTIMIENTO:\n{state['sentimiento']}\n\n"
        f"RESUMEN:\n{state['resumen']}\n\n"
        f"TEMAS CLAVE:\n{state['temas']}"
    )
    return {"reporte": reporte}


# --- Grafo ------------------------------------------------------------------
def construir_grafo():
    graph = StateGraph(AgentState)
    graph.add_node("analizar_sentimiento", analizar_sentimiento)
    graph.add_node("resumir", resumir)
    graph.add_node("extraer_temas", extraer_temas)
    graph.add_node("agregador", agregador)

    # Fan-out: varias aristas desde START = las 3 ramas corren en paralelo.
    graph.add_edge(START, "analizar_sentimiento")
    graph.add_edge(START, "resumir")
    graph.add_edge(START, "extraer_temas")

    # Fan-in: el agregador espera a que terminen las 3.
    graph.add_edge("analizar_sentimiento", "agregador")
    graph.add_edge("resumir", "agregador")
    graph.add_edge("extraer_temas", "agregador")

    graph.add_edge("agregador", END)
    return graph.compile()


app = construir_grafo()


if __name__ == "__main__":
    resena = sys.argv[1] if len(sys.argv) > 1 else (
        "El envío tardó 10 días y nadie respondía en soporte, "
        "pero el producto es excelente y la calidad superó lo que esperaba."
    )
    resultado = app.invoke({"input": resena})
    print(resultado["reporte"])
