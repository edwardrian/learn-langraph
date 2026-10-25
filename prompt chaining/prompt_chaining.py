r"""
Workflow: Prompt Chaining con LangGraph + prompts de LangSmith.

Cada paso es una llamada a un prompt guardado en LangSmith (Prompt Hub).
El modelo y su proveedor se configuran en LangSmith junto al prompt;
aquí solo usamos el identificador del prompt.

Flujo:
    START -> generar -> [gate] --Pass--> mejorar -> pulir -> END
                              \--Fail--> END

Ejecutar (desde la raíz del repo):
    .venv/bin/python "prompt chaining/prompt_chaining.py" "gatos"
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langsmith import Client

# Un solo .env en la raíz del repo, compartido por todos los temas
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("LANGSMITH_PROJECT", "prompt-chaining")  # trazas separadas por tema

# Identificadores de los prompts en LangSmith (Prompt Hub).
# Formato: "nombre" o "nombre:commit" para fijar una versión concreta.
PROMPT_GENERAR = os.getenv("PROMPT_GENERAR", "pc-generar")  # variables: {topic}
PROMPT_MEJORAR = os.getenv("PROMPT_MEJORAR", "pc-mejorar")  # variables: {texto}
PROMPT_PULIR = os.getenv("PROMPT_PULIR", "pc-pulir")        # variables: {texto}


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
    return respuesta.text if hasattr(respuesta, "text") else str(respuesta)


# --- Estado -----------------------------------------------------------------
class AgentState(TypedDict, total=False):
    topic: str        # entrada del usuario
    borrador: str     # salida del paso 1 (generar)
    mejorado: str     # salida del paso 2 (mejorar)
    final: str        # salida del paso 3 (pulir)


# --- Nodos ------------------------------------------------------------------
def generar(state: AgentState) -> AgentState:
    # Entra: topic  ->  Sale: borrador
    return {"borrador": ejecutar_prompt(PROMPT_GENERAR, {"topic": state["topic"]})}


def mejorar(state: AgentState) -> AgentState:
    # Entra: borrador  ->  Sale: mejorado
    return {"mejorado": ejecutar_prompt(PROMPT_MEJORAR, {"texto": state["borrador"]})}


def pulir(state: AgentState) -> AgentState:
    # Entra: mejorado  ->  Sale: final
    return {"final": ejecutar_prompt(PROMPT_PULIR, {"texto": state["mejorado"]})}


# --- Gate -------------------------------------------------------------------
def gate(state: AgentState) -> str:
    """Chequeo programático (sin LLM) sobre la salida del paso 1.

    Pass -> el borrador sigue en la cadena.
    Fail -> se corta el flujo y no se gastan más llamadas.
    """
    borrador = state.get("borrador", "").strip()
    palabras = len(borrador.split())
    if 5 <= palabras <= 300:
        return "Pass"
    return "Fail"


# --- Grafo ------------------------------------------------------------------
def construir_grafo():
    graph = StateGraph(AgentState)
    graph.add_node("generar", generar)
    graph.add_node("mejorar", mejorar)
    graph.add_node("pulir", pulir)

    graph.add_edge(START, "generar")
    graph.add_conditional_edges("generar", gate, {"Pass": "mejorar", "Fail": END})
    graph.add_edge("mejorar", "pulir")
    graph.add_edge("pulir", END)
    return graph.compile()


app = construir_grafo()


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "gatos"
    resultado = app.invoke({"topic": topic})

    print("== Borrador ==\n", resultado.get("borrador"))
    if "final" in resultado:
        print("\n== Mejorado ==\n", resultado["mejorado"])
        print("\n== Final ==\n", resultado["final"])
    else:
        print("\nEl gate falló: el flujo se cortó después del paso 1.")
