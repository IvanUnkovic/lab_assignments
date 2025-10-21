from collections import deque
from typing import List, Dict, Tuple
import copy

Graph = Dict[str, List[str]]
Circuit = List[str]
EulerPath = List[str]

def augmented_hierholzer(G: Graph, start: str) -> Tuple[EulerPath, List[Circuit]]:
    """
    Args:
        G (Graph): A Graph as an adjacency matrix. Assumed to be Eulerian.
        start (str): Starting node for the Hierholzer algorithm.
    Returns:
        Tuple[EulerPath, List[Circuit]]: A tuple containing an Eulerian path in the Euler graph
        and a list of all the circuits found on the path.
    """
    stack = deque()

    stack.append(start)

    eulers_path = []
    graph_circuits = []
    
    while stack:
        u = stack[-1]
        adj = G[u]
        if len(adj) > 0:
            print("ima", stack)
            v = G[u][0]
            stack.append(v)
            G[u].remove(v)
            G[v].remove(u)
        else:
            eulers_path.append(stack.pop())

    nodes = list(G.keys())
    for start_node in nodes:
        graph_circuit = [start_node]
        first = eulers_path.index(start_node)
        j = first+1
        while j is not first:
            graph_circuit.append(eulers_path[j])
            if eulers_path[j] is start_node:
                graph_circuits.append(graph_circuit)
                break
            eulers_path_length = len(eulers_path)
            j = (j + 1) % eulers_path_length

    return eulers_path, graph_circuits