r"""
Workflow: Routing con LangGraph + prompts de LangSmith.

Un prompt "router" clasifica la consulta y LangGraph la envía a UNA sola rama
especializada. Cada rama es otro prompt guardado en LangSmith (Prompt Hub).
El modelo y su proveedor se configuran en LangSmith junto al prompt;
aquí solo usamos el identificador del prompt.

Flujo:
                     /--tecnico-->   soporte_tecnico  --\
    START -> router ----reembolso--> reembolsos       ----> END
                     \--general-->   preguntas_generales/

Ejecutar:
    python routing.py "No me llega el código de verificación"
"""

import os
import sys
from functools import lru_cache
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langsmith import Client

load_dotenv()

# Identificadores de los prompts en LangSmith (Prompt Hub).
# Formato: "nombre" o "nombre:commit" para fijar una versión concreta.
PROMPT_ROUTER = os.getenv("PROMPT_ROUTER", "rt-router")        # variables: {input}
PROMPT_TECNICO = os.getenv("PROMPT_TECNICO", "rt-tecnico")     # variables: {input}
PROMPT_REEMBOLSO = os.getenv("PROMPT_REEMBOLSO", "rt-reembolso")  # variables: {input}
PROMPT_GENERAL = os.getenv("PROMPT_GENERAL", "rt-general")     # variables: {input}

Categoria = Literal["tecnico", "reembolso", "general"]
CATEGORIAS = ("tecnico", "reembolso", "general")


@lru_cache
def cargar_prompt(prompt_id: str):
    """Descarga el prompt desde LangSmith junto con su modelo configurado.

    include_model=True devuelve un Runnable (prompt | modelo), listo para .invoke().
    Se cachea para no volver a descargarlo en cada llamada.
    """
    return Client().pull_prompt(prompt_id, include_model=True)


def ejecutar_prompt(prompt_id: str, variables: dict):
    """Invoca el prompt y devuelve la respuesta tal cual (mensaje o dict)."""
    return cargar_prompt(prompt_id).invoke(variables)


def a_texto(respuesta) -> str:
    return respuesta.content if hasattr(respuesta, "content") else str(respuesta)


# --- Estado -----------------------------------------------------------------
class AgentState(TypedDict, total=False):
    input: str            # consulta del cliente
    categoria: Categoria  # decisión del router
    output: str           # respuesta de la rama elegida


# --- Nodo router ------------------------------------------------------------
def router(state: AgentState) -> AgentState:
    """Clasifica la consulta. Entra: input  ->  Sale: categoria.

    Acepta dos formas de salida del prompt en LangSmith:
    - Con salida estructurada (schema con campo "categoria"): llega un dict.
    - Sin schema: llega un mensaje de texto con la categoría.
    """
    respuesta = ejecutar_prompt(PROMPT_ROUTER, {"input": state["input"]})
    if isinstance(respuesta, dict):
        valor = str(respuesta.get("categoria", ""))
    else:
        valor = a_texto(respuesta)

    valor = valor.strip().lower().replace("é", "e")
    categoria = next((c for c in CATEGORIAS if c in valor), "general")
    return {"categoria": categoria}


# --- Ramas especializadas ---------------------------------------------------
def soporte_tecnico(state: AgentState) -> AgentState:
    # Entra: input  ->  Sale: output
    return {"output": a_texto(ejecutar_prompt(PROMPT_TECNICO, {"input": state["input"]}))}


def reembolsos(state: AgentState) -> AgentState:
    return {"output": a_texto(ejecutar_prompt(PROMPT_REEMBOLSO, {"input": state["input"]}))}


def preguntas_generales(state: AgentState) -> AgentState:
    return {"output": a_texto(ejecutar_prompt(PROMPT_GENERAL, {"input": state["input"]}))}


# --- Decisión condicional ---------------------------------------------------
def route_decision(state: AgentState) -> Categoria:
    """Lee la categoría que dejó el router y elige la rama. Sin LLM."""
    return state["categoria"]


# --- Grafo ------------------------------------------------------------------
def construir_grafo():
    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("soporte_tecnico", soporte_tecnico)
    graph.add_node("reembolsos", reembolsos)
    graph.add_node("preguntas_generales", preguntas_generales)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "tecnico": "soporte_tecnico",
            "reembolso": "reembolsos",
            "general": "preguntas_generales",
        },
    )
    # Solo se ejecuta UNA rama; todas terminan en END.
    graph.add_edge("soporte_tecnico", END)
    graph.add_edge("reembolsos", END)
    graph.add_edge("preguntas_generales", END)
    return graph.compile()


app = construir_grafo()


if __name__ == "__main__":
    consulta = sys.argv[1] if len(sys.argv) > 1 else "Quiero que me devuelvan el dinero de mi pedido"
    resultado = app.invoke({"input": consulta})

    print("== Categoría ==\n", resultado["categoria"])
    print("\n== Respuesta ==\n", resultado["output"])
