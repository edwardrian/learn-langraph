# Laboratorio de patrones de agentes con LangGraph

Espacio para aprender, **un tema por carpeta**, los patrones de workflows y agentes
(Prompt chaining, Routing, Parallelization…). Cada tema tiene:

- **Código** en LangGraph (`<tema>/<tema>.py`), ejecutable desde la terminal o en LangGraph Studio.
- **Prompts y modelo** guardados en LangSmith (Prompt Hub): el código solo usa su identificador.
- **Diagrama explicativo** en Excalidraw+ (no se guarda en el repo).

> Analogía: cada carpeta es una "receta" distinta de cocina con agentes. El código es
> la receta, LangSmith es la despensa donde están los ingredientes (prompts y modelo)
> y Excalidraw es la foto del plato terminado.

## Temas

| Tema | Carpeta | Qué hace el ejemplo | Diagrama |
|---|---|---|---|
| Workflow: Prompt chaining | [`prompt chaining/`](prompt%20chaining/prompt_chaining.py) | Genera un texto, lo valida con un gate y lo mejora en pasos encadenados | [Excalidraw+](https://app.excalidraw.com/s/QkmJ8Er8OV/9LrCOKuF1dF) |
| Workflow: Routing | [`routing/`](routing/routing.py) | Atención al cliente de vehículos: clasifica la consulta (reparación, reclamo o producto) y la envía a una sola rama | [Excalidraw+](https://app.excalidraw.com/s/QkmJ8Er8OV/7uHc5d51cZO) |
| Workflow: Parallelization | [`parallelization/`](parallelization/parallelization.py) | Cotización de repuestos: un gate rápido revisa que haya ítems y ciudad; luego inventario, logística, pricing y recomendaciones en paralelo, y un sintetizador que arma la cotización. Incluye memoria de conversación (checkpointer) y memoria del cliente entre conversaciones (store) | [Excalidraw+](https://app.excalidraw.com/s/QkmJ8Er8OV/6AUr7FzfDWE) |

Próximos temas: Orchestrator-workers → Evaluator-optimizer → Agent (bucle con herramientas).

### Flujos

```
Prompt chaining:  START -> generar -> [gate] --Pass--> mejorar -> pulir -> END
                                            \--Fail--> END

Routing:                             /--reparacion--> reparaciones --\
                  START -> router ----reclamo-----> reclamos     ----> END
                                     \--producto----> productos    --/

Parallelization:  START -> armar_solicitud -> validar_solicitud -> [gate]
                  [gate] --incompleta--> pedir_datos ---------------------------------> guardar_memoria -> END
                  [gate] --apta--> repartir -> en paralelo: inventario | logistica | pricing | recomendaciones -> consolidar
                  consolidar --completo--> redactar_cotizacion -----------------------> guardar_memoria -> END
                  consolidar --incompleto--> cotizacion_incompleta -------------------> guardar_memoria -> END
```

## Estructura

```
c-langraph-test/
├── README.md                 ← este archivo
├── CLAUDE.md                 ← reglas del proyecto para Claude Code
├── requirements.txt          ← dependencias compartidas
├── .env.example              ← plantilla del .env (una sección por tema)
├── langgraph.json            ← grafos registrados para LangGraph Studio
├── .mcp.json                 ← servidores MCP (LangSmith y Excalidraw+), sin secretos
├── langraph-test/            ← laboratorio base (notebooks)
├── prompt chaining/
│   └── prompt_chaining.py
├── routing/
│   └── routing.py
└── parallelization/
    └── parallelization.py
```

## Puesta en marcha

### 1. Entorno virtual

Un solo `.venv` para todos los temas (requiere [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

### 2. Variables de entorno

```bash
cp .env.example .env
```

Rellena en `.env`:

| Variable | Para qué |
|---|---|
| `LANGSMITH_API_KEY` | Descargar los prompts de LangSmith y enviar las trazas |
| `LANGSMITH_TRACING=true` | Activar las trazas de cada nodo |
| `GOOGLE_API_KEY` o `GEMINI_API_KEY` | El modelo de los prompts (hoy todos usan `gemini-3.1-pro-preview`) |
| `PROMPT_<NOMBRE>` | Identificador de cada prompt. Opcional: el código ya trae el nombre por defecto. Usa `nombre:commit` para fijar una versión |

El `.env` nunca se sube a git; solo se sube `.env.example`.

## Ejecutar

Desde la raíz del repo:

```bash
.venv/bin/python "prompt chaining/prompt_chaining.py" "gatos"
.venv/bin/python routing/routing.py "Mi auto hace un ruido raro al frenar"
.venv/bin/python parallelization/parallelization.py "Hola, quiero cotizar unos productos"
```

Parallelization abre un chat con dos memorias en RAM (se pierden al cerrar el script):

| Memoria | Componente | Qué recuerda |
|---|---|---|
| Corto plazo | `InMemorySaver` + `thread_id` | Los mensajes de la conversación: puedes dar los datos en varios mensajes |
| Largo plazo | `InMemoryStore` + `cliente_id` (context) | La ficha del cliente (nombre, ciudad, vehículo) y su última cotización, en `("clientes", cliente_id)` |

Escribe `/nuevo` para abrir otra conversación del mismo cliente (ya sabe su ciudad y
vehículo) y Enter vacío para salir.

### Ver los grafos en LangGraph Studio

```bash
.venv/bin/langgraph dev --port 2025
```

Abre https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2025 y elige el
grafo arriba a la izquierda. Se detiene con `Ctrl+C`.

- Studio guarda sus threads en `.langgraph_api/` (ignorado por git; se puede borrar).
- Studio pone su propia persistencia y store: los grafos que exporta cada script (`app`)
  se compilan **sin** checkpointer ni store. Lo que guarda el store se ve en la pestaña
  **Memory**; en parallelization, indica el `cliente_id` en el context del asistente.
- Si ves `Thread or assistant not found`, revisa que la URL use el puerto **2025**
  (el 2024 es el puerto por defecto de Studio) o abre un thread nuevo.

## Convenciones del código

- API moderna de `StateGraph`, con el estado explícito en un `AgentState` (`TypedDict`).
- **Los prompts y el modelo viven en LangSmith.** El código solo hace
  `Client().pull_prompt(PROMPT_ID, include_model=True).invoke(variables)`; no hay
  código de un proveedor concreto. Para cambiar de modelo, se sube un commit nuevo
  del prompt en LangSmith.
- Cada tema usa un prefijo en sus prompts para que no choquen: `pc-`, `rt-`, `par-`.
- Las decisiones (gate, router, scatter, integridad) son **funciones normales sin LLM**
  dentro de `add_conditional_edges`.
  En parallelization, el gate va **antes** del fan-out: si faltan ítems o ciudad, se piden
  al cliente sin lanzar los workers (1 llamada barata en vez de 6).
- Cada nodo lleva un comentario breve de qué entra al estado y qué sale.
- Cada script carga el `.env` de la raíz y fija su propio proyecto de trazas
  (`LANGSMITH_PROJECT`, por defecto el nombre del tema).

### Probar sin gastar tokens

Reemplaza `cargar_prompt` por un `RunnableLambda` falso y ejecuta el grafo:

```python
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import importlib.util

spec = importlib.util.spec_from_file_location("par", "parallelization/parallelization.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

m.cargar_prompt = lambda pid: RunnableLambda(lambda v: AIMessage(f"<{pid}>"))
print(m.app.invoke({"messages": [("human", "4 llantas 205/55 R16, envío a Cuenca")]})["output"])
```

## Agregar un tema nuevo

1. Crea `<tema>/<tema>.py` en la raíz (un tema por carpeta, sin mezclar).
2. Crea sus prompts en LangSmith con un prefijo propio y agrega sus variables
   `PROMPT_<NOMBRE>` como sección nueva en `.env.example`.
3. Si necesita dependencias nuevas, agrégalas a `requirements.txt`.
4. Registra su `app` en `langgraph.json` para verlo en Studio.
5. Dibuja su diagrama en Excalidraw+ (colección **agents**) y agrega su fila a la
   tabla de temas de este README y del `CLAUDE.md`.

## Herramientas MCP (Claude Code)

`.mcp.json` conecta Claude Code con:

- **LangSmith** (`https://api.smith.langchain.com/mcp`): revisar prompts y trazas.
- **Excalidraw+** (`https://api.excalidraw.com/api/v1/mcp`): crear los diagramas.

Las API keys se leen de variables del shell (`LANGSMITH_API_KEY`, `EXCALIDRAW_API_KEY`
en `~/.zshrc`), así que el archivo no lleva secretos.
