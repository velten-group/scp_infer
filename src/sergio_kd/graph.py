import numpy as np 
import networkx as nx

class DirectedGraphGenerator:
    """
    -------------------------------------------------------------------
    Create the structure of a Directed (potentially cyclic) graph
    -------------------------------------------------------------------
    Args:
    nodes (int)            : Number of nodes in the graph.
    expected_density (int) : Expected number of edges per node.
    """

    def __init__ (self, nodes=30, expected_density=3, enforce_dag=False):
        self.nodes = nodes
        self.expected_density = expected_density  
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
        self.p_node = expected_density/nodes
        self.cyclic = None
        self.enforce_dag = enforce_dag

    def __call__(self):
        vertices = np.arange(self.nodes)
        n_edges = int(np.round(self.expected_density * self.nodes))

        # 1. get all the allowed edges
        allowed_edges = []
        for i in range(self.nodes):
            if self.enforce_dag:
                possible_parents = vertices[:i]
            else:
                possible_parents = np.setdiff1d(vertices, i)
            allowed_edges += [[parent, i] for parent in possible_parents]
        
        assert n_edges <= len(allowed_edges), 'edge density is higher than allowed: {} > {}'.format(self.expected_density, len(allowed_edges)/self.nodes)

        # 2. randomly select n_edges from the allowed edges
        edge_inds = np.random.choice(np.arange(len(allowed_edges)), size=n_edges, replace=False)
        edges = [allowed_edges[ind] for ind in edge_inds]
        # 3. update the adjacency matrix
        for edge in edges:
            self.adjacency_matrix[edge[0]][edge[1]] = 1

        self.g = nx.DiGraph(self.adjacency_matrix)
        self.cyclic = not nx.is_directed_acyclic_graph(self.g)
        return self.g