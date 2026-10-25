from typing import TypedDict
from langgraph.graph import StateGraph, langgraph, START, END


class AgentState(TypedDict):
    name: int
    operation: str
    number: int
    finalnumber: int



def added(state: AgentState) -> AgentState: 
    state['finalnumber'] = state['name'] + state['number']
    return state


def subtrator(state: AgentState) -> AgentState:
    return state



def next(state: AgentState) -> AgentState:

    if state['operation'] == '+':
        return 'added' 
    
    if state['operation'] == '-':
        return 'subtrator'


graph = StateGraph(AgentState)  
graph.add_node('added',added)
graph.add_node('subtrator',subtrator)
graph.add_node('router', lambda state:state)

graph.add_edge(START, 'router')

app = graph.compile()


from IPython.display import Image, display
display(Image,(app.get_graph()).draw_mermaid_png())