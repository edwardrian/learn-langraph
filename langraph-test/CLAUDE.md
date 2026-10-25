# Contexto: Laboratorio de Estudio de LangGraph

Este repositorio es un entorno de práctica técnica para aprender y experimentar con flujos de agentes y orquestación cíclica usando **LangGraph**.

## Comandos del Entorno

- Activar entorno virtual: `source .venv/bin/activate` o `poetry shell`
- Instalar dependencias: `pip install -r requirements.txt` (o `poetry install`)
- Ejecutar un grafo/script: `python -m src.agents.<nombre_archivo>`
- Ejecutar pruebas: `pytest -v`
- Formateo / Linting: `ruff check .` / `black .`

## Reglas de Arquitectura y LangGraph

- **Paradigma de Grafo:** Usar siempre la API moderna de `StateGraph`. No usar cadenas heredadas de LangChain (`LLMChain`, `AgentExecutor`, etc.).
- **Manejo del Estado (`TypedDict` o Pydantic):**
  - Cada grafo debe definir su estado explícitamente (`AgentState`).
  - Utilizar anotaciones de reducción (`Annotated[list, add_messages]`) cuando se maneje historial de chat o mensajes para evitar sobrescrituras accidentales.
- **Estructura de Nodos:**
  - Cada nodo debe ser una función pura o método que recibe el estado y retorna una mutación o diccionario parcial del estado.
  - Separar claramente la invocación del LLM, la ejecución de herramientas (`ToolNode`) y la lógica de decisión condicional (`conditional_edges`).
- **Persistencia & Human-in-the-Loop:**
  - Al implementar checkpoints de memoria, usar `MemorySaver` o checkpointers de base de datos (`SqliteSaver`).
  - Explicar o documentar cuándo se interrumpe el grafo (`interrupt_before` / `interrupt_after`).
- **Enfoque Pedagógico:**
  - Al generar o refactorizar nodos, incluir breves comentarios explicando qué entra al estado y qué sale de él.
