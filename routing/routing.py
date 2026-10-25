r"""
Workflow: Routing con LangGraph + prompts de LangSmith.

Caso: atención al cliente de un concesionario/taller de vehículos.
Un prompt "router" clasifica la consulta del cliente y LangGraph la envía a UNA
sola rama especializada: reparaciones, reclamos o productos. Cada rama es otro prompt guardado en LangSmith (Prompt Hub).
El modelo y su proveedor se configuran en LangSmith junto al prompt;
aquí solo usamos el identificador del prompt.

Flujo:
                     /--reparacion--> reparaciones --\
    START -> router ----reclamo-----> reclamos     ----> END
                     \--producto----> productos    --/

Ejecutar (desde la raíz del repo):
    .venv/bin/python routing/routing.py "Mi auto hace un ruido raro al frenar"
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langsmith import Client

# Un solo .env en la raíz del repo, compartido por todos los temas
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("LANGSMITH_PROJECT", "routing")  # trazas separadas por tema

# Identificadores de los prompts en LangSmith (Prompt Hub).
# Formato: "nombre" o "nombre:commit" para fijar una versión concreta.
PROMPT_ROUTER = os.getenv("PROMPT_ROUTER", "rt-router")        # variables: {input}
PROMPT_REPARACION = os.getenv("PROMPT_REPARACION", "rt-reparacion")  # variables: {input}
PROMPT_RECLAMO = os.getenv("PROMPT_RECLAMO", "rt-reclamo")           # variables: {input}
PROMPT_PRODUCTO = os.getenv("PROMPT_PRODUCTO", "rt-producto")        # variables: {input}

# reparacion: fallas, mantenimiento, citas de taller, garantía mecánica
# reclamo:    quejas por un servicio o compra ya realizados
# producto:   repuestos, accesorios, vehículos o servicios que el cliente quiere adquirir
Categoria = Literal["reparacion", "reclamo", "producto"]
CATEGORIAS = ("reparacion", "reclamo", "producto")


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
    return respuesta.text if hasattr(respuesta, "text") else str(respuesta)


# --- Estado -----------------------------------------------------------------
class AgentState(TypedDict, total=False):
    input: str            # consulta del cliente sobre su vehículo
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

    valor = valor.strip().lower().replace("ó", "o")
    # Si el router responde algo inesperado, se trata como consulta de producto.
    categoria = next((c for c in CATEGORIAS if c in valor), "producto")
    return {"categoria": categoria}


# --- Ramas especializadas ---------------------------------------------------
def reparaciones(state: AgentState) -> AgentState:
    # Entra: input (falla o mantenimiento)  ->  Sale: output (diagnóstico inicial / cita)
    return {"output": a_texto(ejecutar_prompt(PROMPT_REPARACION, {"input": state["input"]}))}


def reclamos(state: AgentState) -> AgentState:
    # Entra: input (queja)  ->  Sale: output (disculpa + pasos para resolverla)
    return {"output": a_texto(ejecutar_prompt(PROMPT_RECLAMO, {"input": state["input"]}))}


def productos(state: AgentState) -> AgentState:
    # Entra: input (lo que necesita)  ->  Sale: output (opciones y disponibilidad)
    return {"output": a_texto(ejecutar_prompt(PROMPT_PRODUCTO, {"input": state["input"]}))}


# --- Decisión condicional ---------------------------------------------------
def route_decision(state: AgentState) -> Categoria:
    """Lee la categoría que dejó el router y elige la rama. Sin LLM."""
    return state["categoria"]


# --- Grafo ------------------------------------------------------------------
def construir_grafo():
    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("reparaciones", reparaciones)
    graph.add_node("reclamos", reclamos)
    graph.add_node("productos", productos)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "reparacion": "reparaciones",
            "reclamo": "reclamos",
            "producto": "productos",
        },
    )
    # Solo se ejecuta UNA rama; todas terminan en END.
    graph.add_edge("reparaciones", END)
    graph.add_edge("reclamos", END)
    graph.add_edge("productos", END)
    return graph.compile()


app = construir_grafo()


if __name__ == "__main__":
    consulta = sys.argv[1] if len(sys.argv) > 1 else "Necesito cotizar llantas para mi camioneta"
    resultado = app.invoke({"input": consulta})

    print("== Categoría ==\n", resultado["categoria"])
    print("\n== Respuesta ==\n", resultado["output"])
