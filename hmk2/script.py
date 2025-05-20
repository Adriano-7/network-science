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
    

class NormalizedPageRank:
    """
    Calculates the PageRank scores of a graph using the power iteration method.
    PageRank values of all nodes are stored in the page_rank attribute as a numpy array.
    Damping factor is represented by Beta.
    """
    def __init__(self, graph: Graph, beta: float = 0.85, convergence_threshold: float = 1e-6):
        self.graph = graph
        self.beta = beta
        self.n = graph.n
        self.page_rank = np.ones(self.n) / self.n
        self.convergence_threshold = convergence_threshold
        self.page_rank_history = [self.page_rank.copy()]

    def calculate_scores(self):
        while True:
            new_page_rank = np.zeros(self.n)
            for i in range(self.n):
                neighbors = self.graph._get_neighbors(i)
                if neighbors:
                    for neighbor in neighbors:
                        new_page_rank[neighbor] += self.page_rank[i] / len(neighbors)
            new_page_rank = (1 - self.beta) / self.n + self.beta * new_page_rank
            if np.linalg.norm(new_page_rank - self.page_rank, ord=1) < self.convergence_threshold:
                break
            self.page_rank = new_page_rank
            self.page_rank_history.append(self.page_rank.copy())

    def get_top_k_nodes(self, k: int):
        return np.argsort(self.page_rank)[-k:][::-1]
    
    def get_score_of_node(self, node: int):
        return self.page_rank[node]
    
    def reset(self):
        self.page_rank = np.ones(self.n) / self.n
        self.page_rank_history = [self.page_rank.copy()]

    def number_of_iterations(self):
        return len(self.page_rank_history)

def exercise_4():
    net4 = Graph(filename='docs/ex4_network.txt')
    pr = NormalizedPageRank(net4)
    pr.calculate_scores()

    # Print pageranks to console
    for i in range(pr.number_of_iterations()):
        print(f"PR state at it. {i}: {pr.page_rank_history[i]}")

    # Plot pagerank values over iterations
    plt.figure(figsize=(10, 6))
    for i in range(pr.n):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        plt.plot(range(pr.number_of_iterations()), [pr.page_rank_history[j][i] for j in range(pr.number_of_iterations())], label=f'Node {alphabet[i]}')
    plt.xlabel('Iterations')
    plt.ylabel('PageRank Value')
    plt.title('PageRank Values Over Iterations')
    plt.legend()
    plt.grid()
    plt.savefig('docs/ex4_pagerank_values.png')
    plt.show()


if __name__ == "__main__":
    exercise_4()