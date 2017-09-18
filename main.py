import ast
import numbers


class WeightedDirectedGraph(object):
    """ Represent a weighted directed graph. Any pair of nodes can be connected with at most one arc. """

    def __init__(self):
        # Use a dictionary of dictionaries as the data structure to represent weighted direct graph.
        # Example: `{'A': {'B': 5, 'C': 1}}` means A -> B has a weight of 5 and A -> C has a weight of 1.
        self.graph_dict = {}

    def add_arc(self, from_node, to_node, weight):
        """
        Add an arc with a weight of `weight` from `from_node` to `to_node` to the graph.
        If there's already an existing arc between the two nodes, its weight will get updated with `weight`.
        """
        if from_node is None or to_node is None:
            return

        if not isinstance(weight, numbers.Number):
            return

        if from_node not in self.graph_dict:
            self.graph_dict[from_node] = {}

        self.graph_dict[from_node][to_node] = weight

    def show(self):
        """
        Return a 3-tuple list representing the graph.
        Example: [('A', 'B', 5), ('A', 'C', 1), ...]
        """
        ret = []
        for start_node, destinations in self.graph_dict.items():
            for end_node, weight in destinations.items():
                ret.append((start_node, end_node, weight))

        return ret

    def compute_path_weight(self, path):
        """
        Return the total weight of a provided path with n number of nodes, denoting by a sequence.
        Example: ('A', 'B', 'C') denotes the path A -> B -> C.
        """
        ret = 0

        # Transform the provided path to a sequence of connections, where each connection is denoted by a 2-tuple.
        # Example: A -> B -> C would transform to [('A', 'B'), ('B', 'C')].
        connections = zip(path, path[1:])

        for conn in connections:
            start_node = conn[0]
            end_node = conn[1]
            try:
                ret += self.graph_dict[start_node][end_node]
            except KeyError:
                return None

        return ret

    def compute_max_stop_count_paths(self, from_node, to_node, max_stop_count):
        """
        Return all possible paths, in a list of tuples, between `from_node` and `to_node`, in which
        the number of stops of each path does not exceed `max_stop_count`.
        Example: [('A', 'B', 'C'), ('A', 'C'), ('A', 'D', 'E', 'C')]
        """
        if max_stop_count <= 0:
            return None

        ret = []
        for neighbour_node in self.graph_dict[from_node].keys():
            if neighbour_node == to_node:
                ret.append((from_node,) + (neighbour_node,))

            # List of paths from the neighbour node to `to_node`; divide-and-conquer the problem.
            neighbour_node_to_dest_paths = self.compute_max_stop_count_paths(neighbour_node,
                                                                             to_node,
                                                                             max_stop_count - 1)
            if neighbour_node_to_dest_paths is None:
                continue

            for sub_path in neighbour_node_to_dest_paths:
                ret.append((from_node,) + sub_path)

        return ret

    def compute_max_weight_paths(self, from_node, to_node, max_weight):
        """
        Return all possible paths, in a list of tuples, between `from_node` and `to_node`, in which
        the weight of each path does not exceed `max_weight`.
        Example: [('A', 'B', 'C'), ('A', 'C'), ('A', 'D', 'E', 'C')]
        """
        if max_weight < 0:
            return None

        ret = []
        for neighbour_node in self.graph_dict[from_node].keys():
            # Remaining weight after travelling to the neighbour node.
            remaining_weight = max_weight - self.graph_dict[from_node][neighbour_node]

            if neighbour_node == to_node and remaining_weight >= 0:
                ret.append((from_node,) + (neighbour_node,))

            # List of paths from the neighbour node to `to_node`; divide-and-conquer the problem.
            neighbour_node_to_dest_paths = self.compute_max_weight_paths(neighbour_node, to_node, remaining_weight)
            if neighbour_node_to_dest_paths is None:
                continue

            for sub_path in neighbour_node_to_dest_paths:
                ret.append((from_node,) + sub_path)

        return ret

    def compute_shortest_length(self, from_node, to_node):
        """
        Return the length (ie. weight) of the shortest path (ie. path with smallest weight)
        between `from_node` and `to_node`.
        """

        # When `from_node` and `to_node` are the same node, the graph needs to be massaged before
        # passing the graph to Dijkstra's algorithm. Manipulate the graph so that `from_node` and `to_node`
        # are viewed as two separate nodes.
        if from_node == to_node:
            graph_dict = self.graph_dict.copy()
            # Annotate `from_node` with a prefix, so that `from_node` and `to_node` are distinguishable
            from_node = '*' + from_node
            graph_dict[from_node] = graph_dict[to_node]
            # `to_node` is an abstract node that has no outoing path.
            graph_dict[to_node] = {}
        else:
            graph_dict = self.graph_dict

        # Keep track of nodes that haven't been visited by the algorithm.
        unvisited_nodes = set(graph_dict.keys())

        # Keep track of the tentative weight from `from_node` to other nodes;
        # `None` means the weight is infinity (ie. unreachable).
        tentative_weight_lookup = {node: None if node != from_node else 0 for node in graph_dict.keys()}

        while len(unvisited_nodes) > 0:
            reachable_unvisited_nodes = dict(filter(lambda x: x[1] is not None and x[0] in unvisited_nodes,
                                                    tentative_weight_lookup.items()))

            if len(reachable_unvisited_nodes) == 0:
                break

            # Find the node with the smallest tentative weight from those unvisited and yet reachable nodes and
            # assign it as the current node.
            current_node = min(reachable_unvisited_nodes, key=reachable_unvisited_nodes.get)

            # Update `tentative_weight_lookup` if there are better alternative paths
            for neighbour_node, weight in graph_dict[current_node].items():
                existing_tentative_weight = tentative_weight_lookup[neighbour_node]
                alt_tentative_weight = weight + tentative_weight_lookup[current_node]

                if existing_tentative_weight is None or alt_tentative_weight < existing_tentative_weight:
                    tentative_weight_lookup[neighbour_node] = alt_tentative_weight

            unvisited_nodes.remove(current_node)

        return tentative_weight_lookup[to_node]


if __name__ == "__main__":
    # Provided arc inputs in short form
    arc_input_list = ['AB5', 'BC4', 'CD8', 'DC8', 'DE6', 'AD5', 'CE2', 'EB3', 'AE7']

    # Construct a graph and initialize it with provided inputs.
    graph = WeightedDirectedGraph()
    for arc_input in arc_input_list:
        graph.add_arc(arc_input[0], arc_input[1], ast.literal_eval(arc_input[2]))

    print('Output #1: {}'.format(graph.compute_path_weight(('A', 'B', 'C')) or 'NO SUCH ROUTE'))
    print('Output #2: {}'.format(graph.compute_path_weight(('A', 'D')) or 'NO SUCH ROUTE'))
    print('Output #3: {}'.format(graph.compute_path_weight(('A', 'D', 'C')) or 'NO SUCH ROUTE'))
    print('Output #4: {}'.format(graph.compute_path_weight(('A', 'E', 'B', 'C', 'D')) or 'NO SUCH ROUTE'))
    print('Output #5: {}'.format(graph.compute_path_weight(('A', 'E', 'D')) or 'NO SUCH ROUTE'))

    CtoC_max_3_stops_paths = graph.compute_max_stop_count_paths('C', 'C', 3)
    print('Output #6: {}'.format(len(CtoC_max_3_stops_paths)))

    AtoC_max_4_stops_paths = graph.compute_max_stop_count_paths('A', 'C', 4)
    # Route with exact 4 stops has 5 nodes, including the start node and the destination node.
    AtoC_exact_4_stops_paths = list(filter(lambda x: len(x) == 5, AtoC_max_4_stops_paths))
    print('Output #7: {}'.format(len(AtoC_exact_4_stops_paths)))

    print('Output #8: {}'.format(graph.compute_shortest_length('A', 'C')))
    print('Output #9: {}'.format(graph.compute_shortest_length('B', 'B')))

    CtoC_max_30_weight_paths = graph.compute_max_weight_paths('C', 'C', 30)
    # Filter out paths with exact weight of 30, if any.
    CtoC_lt_30_weight_paths = list(filter(lambda x: graph.compute_path_weight(x) < 30, CtoC_max_30_weight_paths))
    print('Output #10: {}'.format(len(CtoC_lt_30_weight_paths)))
