import numpy as np

INF = 10 ** 6 - 1

def wfi(weight_matrix: np.ndarray, start: int, end: int):
    """
    Implementation of the WFI algorithm with recreation of the route between two edges.
    Args:
        weight_matrix (np.ndarray): Weight matrix representation of the graph.
        start (int): Starting node in the route.
        end (int): Ending node in the route.
    Returns:
        np.ndarray: Distance matrix of the WFI algorithm.
        list: List of nodes on the route between starting and ending node including starting
            and ending node.
    """
    distance = np.copy(weight_matrix)
    next = np.full_like(weight_matrix, -1)

    nodes_number = len(weight_matrix)
    for i in range(nodes_number):
        for j in range(nodes_number):
            if distance[i, j] != INF and i != j:
                next[i, j] = j
    for k in range(nodes_number):
        for i in range(nodes_number):
            for j in range(nodes_number):
                if  distance[i, j] > distance[i, k] + distance[k, j]:
                    next[i, j] = next[i, k]
                    distance[i, j] = distance[i, k] + distance[k, j]
                    
    path = []
    current_node = start
    while current_node != end:
        path.append(current_node)
        current_node = next[current_node, end]
    path.append(end)

    return distance, path