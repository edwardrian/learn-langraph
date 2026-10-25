r"""
Workflow: Parallelization con LangGraph + prompts de LangSmith.

Varias llamadas al LLM trabajan AL MISMO TIEMPO sobre el mismo input, cada una
en una subtarea independiente (scatter). Cuando todas terminan, se juntan los
resultados (gather): primero se valida con código, luego un LLM redacta.
Cada llamada es un prompt guardado en LangSmith (Prompt Hub); el modelo y su
proveedor se configuran allí. Aquí solo usamos el identificador del prompt.

Caso: cotizar repuestos y accesorios de vehículos más rápido. En vez de consultar
inventario, envío y precios uno tras otro, los workers lo hacen a la vez.
Los datos (stock, tarifas, reglas de precio) están simulados dentro de cada prompt.

Memoria (corto plazo): la conversación se acumula en `messages`. Con un checkpointer
y un thread_id, el cliente puede dar los datos en varios mensajes
("hola, quiero productos" -> "4 llantas 205/55 R16, envío a Cuenca").
- Terminal: InMemorySaver (en RAM; se pierde al cerrar el script).
- LangGraph Studio: pone su propia persistencia, por eso `app` se compila sin checkpointer.

Flujo:
                                             /-> inventario       -\
                                            +--> logistica        --+
    START -> armar_solicitud -> (scatter) -+--> pricing          --+--> consolidar --completo--> redactar_cotizacion -> END
                                             \-> recomendaciones  -/         |
                                                 (opcional)                  \--incompleto--> cotizacion_incompleta -> END

Ejecutar (desde la raíz del repo); abre un chat, Enter vacío para salir:
    .venv/bin/python parallelization/parallelization.py "Hola, quiero cotizar unos productos"
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import Client

# Un solo .env en la raíz del repo, compartido por todos los temas
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("LANGSMITH_PROJECT", "parallelization")  # trazas separadas por tema

# Identificadores de los prompts en LangSmith (Prompt Hub).
# Formato: "nombre" o "nombre:commit" para fijar una versión concreta.
PROMPT_INVENTARIO = os.getenv("PROMPT_INVENTARIO", "par-inventario")                # variables: {input}
PROMPT_LOGISTICA = os.getenv("PROMPT_LOGISTICA", "par-logistica")                   # variables: {input}
PROMPT_PRICING = os.getenv("PROMPT_PRICING", "par-pricing")                         # variables: {input}
PROMPT_RECOMENDACIONES = os.getenv("PROMPT_RECOMENDACIONES", "par-recomendaciones")  # variables: {input}
PROMPT_COTIZACION = os.getenv("PROMPT_COTIZACION", "par-cotizacion")
# variables de PROMPT_COTIZACION: {input} {inventario} {logistica} {pricing} {recomendaciones}

# Workers que SIEMPRE deben responder para poder cotizar.
OBLIGATORIOS = ("inventario", "logistica", "pricing")


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
# Cada worker escribe en SU PROPIA clave, así las ramas paralelas no se pisan.
class AgentState(TypedDict, total=False):
    # add_messages ACUMULA en vez de reemplazar: es la memoria de la conversación.
    messages: Annotated[list[AnyMessage], add_messages]
    input: str                     # solicitud armada con todos los mensajes del cliente
    incluir_recomendaciones: bool  # activa el worker opcional (por defecto: sí)
    inventario: str                # worker 1: disponibilidad y días hasta tener los ítems
    logistica: str                 # worker 2: costo y días de envío por courier
    pricing: str                   # worker 3: precios con descuentos y subtotal
    recomendaciones: str           # worker 4 (opcional): up-selling / cross-selling
    aviso: str                     # motivo para no cotizar ("" si todo está bien)
    output: str                    # cotización final para el cliente


# --- Memoria -> solicitud ---------------------------------------------------
def armar_solicitud(state: AgentState) -> AgentState:
    """Junta TODOS los mensajes del cliente en una sola solicitud. Sin LLM.

    Entra: messages  ->  Sale: input (y limpia recomendaciones del turno anterior)
    """
    del_cliente = [m.text for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    if len(del_cliente) <= 1:
        solicitud = del_cliente[0] if del_cliente else ""
    else:
        lineas = "\n".join(f"{i}. {texto}" for i, texto in enumerate(del_cliente, 1))
        solicitud = f"Mensajes del cliente, en orden (si se contradicen, vale el más reciente):\n{lineas}"
    return {"input": solicitud, "recomendaciones": ""}


# --- Scatter: qué workers lanzar ------------------------------------------
def scatter(state: AgentState) -> list[str]:
    """Devuelve la lista de workers que corren en paralelo. Sin LLM."""
    workers = ["inventario", "logistica", "pricing"]
    if state.get("incluir_recomendaciones", True):
        workers.append("recomendaciones")
    return workers


# --- Workers en paralelo ----------------------------------------------------
def inventario(state: AgentState) -> AgentState:
    # Entra: input  ->  Sale: inventario (stock por ítem, almacén y días de reposición)
    return {"inventario": ejecutar_prompt(PROMPT_INVENTARIO, {"input": state.get("input", "")})}


def logistica(state: AgentState) -> AgentState:
    # Entra: input (destino y peso de los ítems)  ->  Sale: logistica (opciones de courier)
    return {"logistica": ejecutar_prompt(PROMPT_LOGISTICA, {"input": state.get("input", "")})}


def pricing(state: AgentState) -> AgentState:
    # Entra: input  ->  Sale: pricing (precio, descuento por línea y subtotal)
    return {"pricing": ejecutar_prompt(PROMPT_PRICING, {"input": state.get("input", "")})}


def recomendaciones(state: AgentState) -> AgentState:
    # Entra: input  ->  Sale: recomendaciones (1 a 3 complementos)
    return {"recomendaciones": ejecutar_prompt(PROMPT_RECOMENDACIONES, {"input": state.get("input", "")})}


# --- Gather -----------------------------------------------------------------
def consolidar(state: AgentState) -> AgentState:
    """Valida la integridad de los resultados. Es código normal, sin LLM.

    LangGraph lo ejecuta UNA sola vez, cuando todos los workers lanzados han terminado.
    Entra: inventario, logistica, pricing  ->  Sale: aviso
    """
    # par-inventario responde SIN_ITEMS cuando la solicitud no nombra ningún producto.
    if "SIN_ITEMS" in state.get("inventario", ""):
        return {"aviso": "Para cotizar necesitamos saber qué productos buscas: "
                         "dinos los ítems, las cantidades y la ciudad de envío."}
    faltantes = [w for w in OBLIGATORIOS if not state.get(w, "").strip()]
    if faltantes:
        return {"aviso": f"No se pudo cotizar: faltan los resultados de {', '.join(faltantes)}."}
    return {"aviso": ""}


def integridad(state: AgentState) -> Literal["completo", "incompleto"]:
    """Decide si hay datos suficientes para cotizar. Sin LLM."""
    return "incompleto" if state.get("aviso") else "completo"


def redactar_cotizacion(state: AgentState) -> AgentState:
    # Entra: input + resultados de los workers  ->  Sale: output + respuesta guardada en messages
    variables = {
        "input": state.get("input", ""),
        "inventario": state.get("inventario", ""),
        "logistica": state.get("logistica", ""),
        "pricing": state.get("pricing", ""),
        "recomendaciones": state.get("recomendaciones") or "Sin recomendaciones.",
    }
    cotizacion = ejecutar_prompt(PROMPT_COTIZACION, variables)
    return {"output": cotizacion, "messages": [AIMessage(cotizacion)]}


def cotizacion_incompleta(state: AgentState) -> AgentState:
    # Entra: aviso  ->  Sale: output + aviso guardado en messages (no se cotiza con datos a medias)
    aviso = state.get("aviso", "")
    return {"output": aviso, "messages": [AIMessage(aviso)]}


# --- Grafo ------------------------------------------------------------------
def construir_grafo(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("armar_solicitud", armar_solicitud)
    graph.add_node("inventario", inventario)
    graph.add_node("logistica", logistica)
    graph.add_node("pricing", pricing)
    graph.add_node("recomendaciones", recomendaciones)
    graph.add_node("consolidar", consolidar)
    graph.add_node("redactar_cotizacion", redactar_cotizacion)
    graph.add_node("cotizacion_incompleta", cotizacion_incompleta)

    graph.add_edge(START, "armar_solicitud")

    # Fan-out: scatter devuelve una LISTA de nodos = todos corren en paralelo.
    graph.add_conditional_edges("armar_solicitud", scatter, ["inventario", "logistica", "pricing", "recomendaciones"])

    # Fan-in: consolidar espera a que terminen los workers lanzados.
    for worker in ("inventario", "logistica", "pricing", "recomendaciones"):
        graph.add_edge(worker, "consolidar")

    graph.add_conditional_edges(
        "consolidar",
        integridad,
        {"completo": "redactar_cotizacion", "incompleto": "cotizacion_incompleta"},
    )
    graph.add_edge("redactar_cotizacion", END)
    graph.add_edge("cotizacion_incompleta", END)
    return graph.compile(checkpointer=checkpointer)


# Studio pone su propia persistencia: si `app` trae un checkpointer, se niega a cargarlo.
app = construir_grafo()


if __name__ == "__main__":
    # Memoria en RAM: el mismo thread_id recuerda los mensajes anteriores.
    chat = construir_grafo(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "cliente-1"}}

    mensaje = sys.argv[1] if len(sys.argv) > 1 else input("Cliente: ")
    while mensaje.strip():
        resultado = chat.invoke({"messages": [HumanMessage(mensaje)]}, config)
        print(f"\nAsesor: {resultado['output']}\n")
        mensaje = input("Cliente (Enter vacío para salir): ")
