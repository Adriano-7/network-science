import numpy as np
import random
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, n: int = None, filename: str = None):
        """
        Initializes a graph with n nodes (with no connections) or loads it from a file.
        Sets all nodes as unvisited.
        """
        if filename:
            with open(filename, 'r') as file:
                first_line = file.readline().strip()
                if first_line:
                    self.n = int(first_line)
                    self.adj_matrix = [[0 for _ in range(self.n)] for _ in range(self.n)]
                for line in file:
                    a, b = map(int, line.strip().split())
                    self.add_edge(a - 1, b - 1)
        elif n is not None:
            self.n = n
            self.adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        else:
            raise ValueError("Either 'n' or 'filename' must be provided.")
        self.visited = [False] * self.n

    def add_edge(self, a : int, b : int):
        self.adj_matrix[a][b] = 1
        self.adj_matrix[b][a] = 1

    def add_edges(self, a : int, targets : list):
        for target in targets:
            self.add_edge(a, target)
 
    def has_edge(self, a, b):
        return self.adj_matrix[a][b] == 1

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.n}\n")
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if self.adj_matrix[i][j] == 1:
                        file.write(f"{i + 1} {j + 1}\n")

    def calculate_giant_component_size(self):
        self._set_all_unvisited()
        max_size = 0
        for i in range(self.n):
            if not self.visited[i]:
                size = self.component_size(i)
                max_size = max(max_size, size)
        return max_size

    def component_size(self, node):
        stack = [node]
        size = 0
        while stack:
            current_node = stack.pop()
            if not self.visited[current_node]:
                self.visited[current_node] = True
                size += 1
                for neighbor in self._get_neighbors(current_node):
                    if not self.visited[neighbor]:
                        stack.append(neighbor)
        return size
    
    def get_degree_of_node(self, node):
        return sum(self.adj_matrix[node])
    
    def _set_all_unvisited(self):
        self.visited = [False] * self.n

    def _get_neighbors(self, node):
        return [i for i in range(self.n) if self.adj_matrix[node][i] == 1]


class ErdosRenyiModel(Graph):
    def __init__(self, n : int, p : float):
        super().__init__(n)
        self.p = p
        self._generate_graph()

    def _generate_graph(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if random.random() < self.p:
                    self.add_edge(i, j)


class BarabasiAlbertModel(Graph):
    def __init__(self, n: int, m0: int, m: int):
        """
        m0: initial number of nodes in a fully connected graph
        m: number of edges to attach from a new node to existing nodes
        n: total number of nodes in the graph
        """
        super().__init__(n)
        self.m0 = m0
        self.m = m
        self._generate_graph()

    def _generate_graph(self):
        # Fully connected graph with m0 nodes
        for i in range(self.m0):
            for j in range(i + 1, self.m0):
                self.add_edge(i, j)

        print(f"Initial graph with {self.m0} nodes created.")

        # Add new nodes one by one
        for i in range(self.m0, self.n):
            targets = self._choose_targets(i)
            self.add_edges(i, targets)
            print(f"Node {i} added with edges to {targets}.")

    def _choose_targets(self, new_node):
        """
        Choose m targets based on the preferential attachment mechanism.
        """
        targets = set()
        while len(targets) < self.m:
            probabilities = [self._probability_of_connection(j) for j in range(self.n)]
            target = random.choices(range(self.n), weights=probabilities)[0]
            if target not in targets and target != new_node:
                targets.add(target)
        print(f"Node {new_node} targets: {targets}")
        return list(targets)
    
    def _probability_of_connection(self, node):
        """
        pi = k_i / sum(k_j)
        """
        degree = self.get_degree_of_node(node)
        return degree / sum(self.get_degree_of_node(j) for j in range(self.n))
    

class EfficientBarabasiAlbertModel(Graph):
    def __init__(self, n: int, m0: int, m: int):
        """
        Efficient implementation of Barabási-Albert model
        
        Parameters:
        - n: total number of nodes in the graph
        - m0: initial number of nodes in a fully connected graph
        - m: number of edges to attach from a new node to existing nodes
        """
        super().__init__(n)
        self.m0 = m0
        self.m = m
        # Track node degrees for efficient preferential attachment
        self.degrees = [0] * n
        # Total number of edges (each edge counted twice, once for each endpoint)
        self.total_edges = 0
        self._generate_graph()
    
    def add_edge(self, a: int, b: int):
        """Override add_edge to update degrees and total_edges"""
        if not self.has_edge(a, b):
            self.adj_matrix[a][b] = 1
            self.adj_matrix[b][a] = 1
            self.degrees[a] += 1
            self.degrees[b] += 1
            self.total_edges += 2
    
    def _generate_graph(self):
        # Fully connected graph with m0 nodes
        for i in range(self.m0):
            for j in range(i + 1, self.m0):
                self.add_edge(i, j)
        
        # Add new nodes one by one
        for i in range(self.m0, self.n):
            targets = self._choose_targets_efficiently(i)
            for target in targets:
                self.add_edge(i, target)
    
    def _choose_targets_efficiently(self, new_node):
        """
        Efficient implementation of preferential attachment using stochastic acceptance
        """
        targets = set()
        max_degree = max(self.degrees[:new_node]) or 1  # Avoid division by zero
        
        while len(targets) < min(self.m, new_node):  # Ensure we don't try to add more edges than possible
            # Select a random node
            candidate = random.randint(0, new_node - 1)
            # Accept with probability proportional to its degree
            if random.random() < self.degrees[candidate] / max_degree:
                if candidate not in targets:
                    targets.add(candidate)
        
        return list(targets)
    
    def get_degree_of_node(self, node):
        """Return the degree directly from our tracking array"""
        return self.degrees[node]


def exercise_4():
    net1 = ErdosRenyiModel(n = 2000, p = 0.0001)
    net2 = ErdosRenyiModel(n = 2000, p = 0.005)
    net1.save_to_file('random1.txt')
    net2.save_to_file('random2.txt')

def exercise_5():
    net1 = Graph(filename='random1.txt')
    net2 = Graph(filename='random2.txt')
    print(f"Size of the giant component in random1.txt: {net1.calculate_giant_component_size()}")
    print(f"Size of the giant component in random2.txt: {net2.calculate_giant_component_size()}")

def exercise_6():
    min_p = 0.0001
    max_p = 0.005
    p_step = 0.0001
    sizes = {}

    while min_p <= max_p:
        net = ErdosRenyiModel(n=2000, p=min_p)
        size = net.calculate_giant_component_size()
        sizes[min_p] = size
        print(f"p: {min_p:.4f}, Size of the giant component: {size}")
        min_p += p_step

    plt.plot(sizes.keys(), sizes.values())
    plt.xlabel('p')
    plt.ylabel('Size of the giant component')
    plt.title('Giant Component Size vs p')
    plt.savefig('docs/ex6.png')
    plt.show()

def exercise_7():
    print("Generating net1")
    net1 = EfficientBarabasiAlbertModel(n=2000, m0=3, m=1)
    print("Generating net2")
    net2 = EfficientBarabasiAlbertModel(n=2000, m0=5, m=2)
    net1.save_to_file('ba1.txt')
    net2.save_to_file('ba2.txt')

def exercise_8():
    """
    8. The previous process should generate a scale-free network with a power law degree distribution with
    exponent α = 3.
    Plot the degree distribution of both your generated networks using cumulative binning (see
    slides 102 and 103) and try to fit with the corresponding power law function (showing it in the plot)
    """
    ba1 = Graph(filename='docs/ba1.txt')
    ba2 = Graph(filename='docs/ba2.txt')
    degrees1 = [ba1.get_degree_of_node(i) for i in range(ba1.n)]
    degrees2 = [ba2.get_degree_of_node(i) for i in range(ba2.n)]

    unique_degrees1, counts1 = np.unique(degrees1, return_counts=True)
    unique_degrees2, counts2 = np.unique(degrees2, return_counts=True)
    density1 = counts1 / sum(counts1)
    density2 = counts2 / sum(counts2)

    alpha = 3
    x_range1 = np.linspace(min(unique_degrees1), max(unique_degrees1), 100)
    x_range2 = np.linspace(min(unique_degrees2), max(unique_degrees2), 100)
    scale_factor1 = density1[0] * (unique_degrees1[0] ** alpha)
    scale_factor2 = density2[0] * (unique_degrees2[0] ** alpha)
    cumulative1 = [sum(density1[j] for j in range(len(density1)) if unique_degrees1[j] >= d) for d in unique_degrees1]
    cumulative2 = [sum(density2[j] for j in range(len(density2)) if unique_degrees2[j] >= d) for d in unique_degrees2]
    cum_power_law1 = scale_factor1 * (x_range1 ** (-alpha+1)) / (alpha-1)
    cum_power_law2 = scale_factor2 * (x_range2 ** (-alpha+1)) / (alpha-1)
    cum_scale1 = cumulative1[0] / cum_power_law1[0]
    cum_scale2 = cumulative2[0] / cum_power_law2[0]

    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(16, 6))

    plot1.scatter(unique_degrees1, cumulative1, label='BA1 Data', color='blue')
    plot1.plot(x_range1, cum_power_law1 * cum_scale1, '--', color='darkblue', label=r'Power Law ($\alpha=3$)')
    plot1.set_xscale('log')
    plot1.set_yscale('log')
    plot1.set_xlabel('Degree (log scale)')
    plot1.set_ylabel('Cumulative Probability (log scale)')
    plot1.set_title('BA1 Network (m0=3, m=1)')
    plot1.legend()
    plot1.grid(True, alpha=0.3)

    plot2.scatter(unique_degrees2, cumulative2, label='BA2 Data', color='red')
    plot2.plot(x_range2, cum_power_law2 * cum_scale2, '--', color='darkred', label=r'Power Law ($\alpha=3$)')
    plot2.set_xscale('log')
    plot2.set_yscale('log')
    plot2.set_xlabel('Degree (log scale)')
    plot2.set_ylabel('Cumulative Probability (log scale)')
    plot2.set_title('BA2 Network (m0=5, m=2)')
    plot2.legend()
    plot2.grid(True, alpha=0.3)

    fig.suptitle('Cumulative Degree Distributions with Power Law Fits (Log-Log Scale)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('docs/ex8.png')
    plt.show()


if __name__ == "__main__":
    exercise_8()
