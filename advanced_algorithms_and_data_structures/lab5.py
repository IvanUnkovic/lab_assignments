import heapq

TOL_DEC = 3
TOLERANCE = 10**-TOL_DEC


class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

    def __lt__(self, other: 'Node') -> bool:
        diff = self.prob - other.prob
        if abs(diff) < TOLERANCE:
            if(self.symbol < other.symbol):
                return True
            else:
                return False
        return True if self.prob + TOLERANCE < other.prob else False
def Huffman_tree(symbol_with_probs: dict) -> Node:
    nodes_queue = [Node(prob, symbol) for symbol, prob in symbol_with_probs.items()]
    while len(nodes_queue) >= 2:
        nodeA = heapq.heappop(nodes_queue)
        nodeB = heapq.heappop(nodes_queue)
        two_probs = nodeA.prob + nodeB.prob
        two_probs_node = Node(two_probs, '', left=nodeA, right=nodeB)
        two_probs_node.left.code = '0'
        two_probs_node.right.code = '1'
        two_probs_node.left.code = two_probs_node.left.code + two_probs_node.code 
        two_probs_node.right.code = two_probs_node.right.code + two_probs_node.code
        heapq.heappush(nodes_queue, two_probs_node)
    return nodes_queue[0]



def roundToDecimals(num: float, decimals: int) -> float:
    return round(num * 10**decimals) / 10**decimals