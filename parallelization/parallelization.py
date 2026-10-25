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

Gate antes del fan-out: un prompt rápido (par-validar) extrae ítems, ciudad y perfil.
Si faltan ítems o ciudad, se le piden al cliente SIN lanzar los workers (no se gastan
4 llamadas caras para responder "¿qué productos buscas?").

Memoria:
- Corto plazo (checkpointer + thread_id): la conversación se acumula en `messages`,
  así el cliente puede dar los datos en varios mensajes.
- Largo plazo (store + cliente_id): la ficha del cliente (nombre, ciudad, vehículo)
  y su última cotización se guardan en ("clientes", cliente_id) y se recuerdan en
  OTRAS conversaciones. El cliente_id llega como context: invoke(..., context={"cliente_id": "ana"}).
- Terminal: InMemorySaver + InMemoryStore (en RAM; se pierden al cerrar el script).
- LangGraph Studio: pone su propia persistencia y store, por eso `app` se compila sin ellos.

Flujo:
    START -> armar_solicitud -> validar_solicitud -> [gate]
    [gate] --incompleta--> pedir_datos ----------------------------------------> guardar_memoria -> END
    [gate] --apta--> repartir -> en paralelo: inventario | logistica | pricing | recomendaciones -> consolidar
    consolidar --completo--> redactar_cotizacion ------------------------------> guardar_memoria -> END
    consolidar --incompleto--> cotizacion_incompleta --------------------------> guardar_memoria -> END

Ejecutar (desde la raíz del repo); abre un chat, /nuevo = otra conversación del mismo cliente,
Enter vacío para salir:
    .venv/bin/python parallelization/parallelization.py "Hola, soy Ana de Cuenca y quiero 4 llantas 205/55 R16"
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
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
PROMPT_VALIDAR = os.getenv("PROMPT_VALIDAR", "par-validar")                         # variables: {input}
PROMPT_COTIZACION = os.getenv("PROMPT_COTIZACION", "par-cotizacion")
# variables de PROMPT_COTIZACION: {input} {inventario} {logistica} {pricing} {recomendaciones}

# Workers que SIEMPRE deben responder para poder cotizar.
OBLIGATORIOS = ("inventario", "logistica", "pricing")
# Campos de la ficha del cliente en el store.
CAMPOS_PERFIL = ("nombre", "ciudad", "vehiculo")


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


# --- Context y estado -------------------------------------------------------
@dataclass
class Contexto:
    """Datos de la ejecución que NO son parte del estado. Sin cliente_id no se usa el store."""
    cliente_id: str = ""


# Cada worker escribe en SU PROPIA clave, así las ramas paralelas no se pisan.
class AgentState(TypedDict, total=False):
    # add_messages ACUMULA en vez de reemplazar: es la memoria de la conversación.
    messages: Annotated[list[AnyMessage], add_messages]
    input: str                     # solicitud armada con los mensajes + la ficha del cliente
    incluir_recomendaciones: bool  # activa el worker opcional (por defecto: sí)
    inventario: str                # worker 1: disponibilidad y días hasta tener los ítems
    logistica: str                 # worker 2: costo y días de envío por courier
    pricing: str                   # worker 3: precios con descuentos y subtotal
    recomendaciones: str           # worker 4 (opcional): up-selling / cross-selling
    items: str                     # gate: ítems pedidos con cantidad ("" si no nombró ninguno)
    perfil: dict                   # gate: nombre, ciudad y vehículo dichos en la conversación
    faltantes: list[str]           # gate: datos que faltan para poder cotizar
    aviso: str                     # motivo para no cotizar ("" si todo está bien)
    output: str                    # cotización final para el cliente


def leer_ficha(runtime: Runtime[Contexto]) -> tuple[tuple[str, str] | None, dict]:
    """Devuelve (namespace, ficha guardada) del cliente, o (None, {}) si no hay store o cliente_id."""
    cliente_id = getattr(runtime.context, "cliente_id", "")
    if runtime.store is None or not cliente_id:
        return None, {}
    namespace = ("clientes", cliente_id)
    item = runtime.store.get(namespace, "perfil")
    return namespace, (item.value if item else {})


# --- Memoria -> solicitud ---------------------------------------------------
def armar_solicitud(state: AgentState, runtime: Runtime[Contexto]) -> AgentState:
    """Junta los mensajes del cliente y su ficha del store en una sola solicitud. Sin LLM.

    Entra: messages (+ store)  ->  Sale: input (y limpia recomendaciones del turno anterior)
    """
    del_cliente = [m.text for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    if len(del_cliente) <= 1:
        solicitud = del_cliente[0] if del_cliente else ""
    else:
        lineas = "\n".join(f"{i}. {texto}" for i, texto in enumerate(del_cliente, 1))
        solicitud = f"Mensajes del cliente, en orden (si se contradicen, vale el más reciente):\n{lineas}"

    _, ficha = leer_ficha(runtime)
    conocidos = "; ".join(f"{campo}: {ficha[campo]}" for campo in CAMPOS_PERFIL if ficha.get(campo))
    if conocidos:
        solicitud += (f"\n\nDatos del cliente de conversaciones anteriores ({conocidos}). "
                      "Úsalos solo si en estos mensajes no dice otra cosa.")
    return {"input": solicitud, "recomendaciones": ""}


# --- Gate: ¿se puede cotizar? ----------------------------------------------
def validar_solicitud(state: AgentState) -> AgentState:
    """Extrae los datos mínimos con un prompt rápido (par-validar), antes del fan-out.

    Entra: input  ->  Sale: items, perfil ({"nombre", "ciudad", "vehiculo"}), faltantes
    """
    texto = ejecutar_prompt(PROMPT_VALIDAR, {"input": state.get("input", "")})
    datos = {}
    for linea in texto.splitlines():
        campo, _, valor = linea.partition(":")
        campo = campo.strip().lower().replace("í", "i")
        if valor.strip() and valor.strip().lower() not in ("vacío", "vacio"):
            datos[campo] = valor.strip()
    perfil = {campo: datos[campo] for campo in CAMPOS_PERFIL if campo in datos}
    faltantes = [dato for dato, campo in (("los productos que buscas (con su cantidad)", "items"),
                                          ("la ciudad de envío", "ciudad")) if campo not in datos]
    return {"items": datos.get("items", ""), "perfil": perfil, "faltantes": faltantes}


def gate(state: AgentState) -> Literal["apta", "incompleta"]:
    """Decide si vale la pena lanzar los workers. Sin LLM."""
    return "incompleta" if state.get("faltantes") else "apta"


def pedir_datos(state: AgentState) -> AgentState:
    # Entra: faltantes  ->  Sale: aviso + output + pregunta guardada en messages (no corre ningún worker)
    aviso = f"Para cotizar necesitamos {' y '.join(state.get('faltantes', []))}."
    return {"aviso": aviso, "output": aviso, "messages": [AIMessage(aviso)]}


# --- Scatter: qué workers lanzar ------------------------------------------
def repartir(state: AgentState) -> AgentState:
    # Nodo vacío: es el punto desde donde sale el fan-out cuando el gate dice "apta".
    return {}


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
    # La falta de ítems o ciudad ya la frenó el gate; aquí solo se revisa que ningún worker falló.
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


# --- Memoria de largo plazo -------------------------------------------------
def guardar_memoria(state: AgentState, runtime: Runtime[Contexto]) -> AgentState:
    """Guarda la ficha del cliente y su última cotización en el store. Sin LLM.

    Entra: perfil, aviso, output  ->  Sale: nada en el estado (escribe en el store)
    """
    namespace, ficha = leer_ficha(runtime)
    if namespace is None or runtime.store is None:
        return {}
    # Lo nuevo de esta conversación reemplaza a lo guardado; lo que no se dijo se conserva.
    runtime.store.put(namespace, "perfil", {**ficha, **state.get("perfil", {})})
    if not state.get("aviso"):
        runtime.store.put(namespace, "ultima_cotizacion", {
            "solicitud": state.get("input", ""),
            "cotizacion": state.get("output", ""),
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    return {}


# --- Grafo ------------------------------------------------------------------
def construir_grafo(checkpointer=None, store=None):
    graph = StateGraph(AgentState, context_schema=Contexto)
    graph.add_node("armar_solicitud", armar_solicitud)
    graph.add_node("validar_solicitud", validar_solicitud)
    graph.add_node("pedir_datos", pedir_datos)
    graph.add_node("repartir", repartir)
    graph.add_node("inventario", inventario)
    graph.add_node("logistica", logistica)
    graph.add_node("pricing", pricing)
    graph.add_node("recomendaciones", recomendaciones)
    graph.add_node("consolidar", consolidar)
    graph.add_node("redactar_cotizacion", redactar_cotizacion)
    graph.add_node("cotizacion_incompleta", cotizacion_incompleta)
    graph.add_node("guardar_memoria", guardar_memoria)

    graph.add_edge(START, "armar_solicitud")
    graph.add_edge("armar_solicitud", "validar_solicitud")

    # Gate: solo si la solicitud es apta se pasa al fan-out.
    graph.add_conditional_edges("validar_solicitud", gate, {"apta": "repartir", "incompleta": "pedir_datos"})

    # Fan-out: scatter devuelve una LISTA de nodos = todos corren en paralelo.
    workers = ["inventario", "logistica", "pricing", "recomendaciones"]
    graph.add_conditional_edges("repartir", scatter, workers)

    # Fan-in: consolidar espera a que terminen los workers lanzados.
    for worker in workers:
        graph.add_edge(worker, "consolidar")

    graph.add_conditional_edges(
        "consolidar",
        integridad,
        {"completo": "redactar_cotizacion", "incompleto": "cotizacion_incompleta"},
    )
    graph.add_edge("redactar_cotizacion", "guardar_memoria")
    graph.add_edge("cotizacion_incompleta", "guardar_memoria")
    graph.add_edge("pedir_datos", "guardar_memoria")
    graph.add_edge("guardar_memoria", END)
    return graph.compile(checkpointer=checkpointer, store=store)


# Studio pone su propia persistencia y store: si `app` trae los suyos, se niega a cargarlo.
app = construir_grafo()


if __name__ == "__main__":
    # Memoria en RAM: el thread_id recuerda la conversación; el store recuerda al cliente.
    chat = construir_grafo(checkpointer=InMemorySaver(), store=InMemoryStore())
    contexto = Contexto(cliente_id="cliente-1")
    conversacion = 1
    config: RunnableConfig = {"configurable": {"thread_id": f"conversacion-{conversacion}"}}

    mensaje = sys.argv[1] if len(sys.argv) > 1 else input("Cliente: ")
    while mensaje.strip():
        if mensaje.strip() == "/nuevo":
            # Otra conversación (thread nuevo) del MISMO cliente: el store sigue ahí.
            conversacion += 1
            config = {"configurable": {"thread_id": f"conversacion-{conversacion}"}}
            print(f"\n-- Conversación {conversacion} (mismo cliente) --\n")
        else:
            resultado = chat.invoke({"messages": [HumanMessage(mensaje)]}, config, context=contexto)
            print(f"\nAsesor: {resultado['output']}\n")
        mensaje = input("Cliente (/nuevo = otra conversación, Enter vacío para salir): ")
