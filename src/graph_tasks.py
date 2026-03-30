import random
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from utils import read_adjacency_list


@dataclass
class TaskParameters:
    """Parameters for different graph analysis tasks"""
    type: str
    params: Dict[str, Any]

    @classmethod
    def create_vertex_degree_params(cls, node: str):
        return cls(type="single_node", params={"node": node})

    @classmethod
    def create_shortest_path_params(cls, start_node: str, end_node: str):
        return cls(type="node_pair", params={
            "start_node": start_node,
            "end_node": end_node
        })

    @classmethod
    def create_common_neighbours_params(cls, start_node: str, end_node: str):
        return cls(type="node_pair", params={
            "start_node": start_node,
            "end_node": end_node
        })


class BaseTask:
    """Base class for graph analysis tasks"""

    def __init__(self, graph_type: str):
        self.graph_type = graph_type

    def generate_stimuli(self, G: nx.Graph, num_stimuli: int, seed: Optional[int] = None) -> List[TaskParameters]:
        """Generate stimuli with optional seed for reproducibility"""
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        raise NotImplementedError

    def compute_ground_truth(self, G: nx.Graph, parameters: TaskParameters) -> Dict:
        raise NotImplementedError


class VertexDegreeTask(BaseTask):
    def generate_stimuli(self, G: nx.Graph, num_stimuli: int, seed: Optional[int] = None) -> List[TaskParameters]:
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        selected_nodes = random.sample(
            list(G.nodes()), min(num_stimuli, len(G)))
        return [TaskParameters.create_vertex_degree_params(node) for node in selected_nodes]

    def compute_ground_truth(self, G: nx.Graph, parameters: TaskParameters) -> Dict:
        node = parameters.params["node"]
        if self.graph_type == "di":
            return {"in_degree": G.in_degree(node), "out_degree": G.out_degree(node)}
        return {"degree": G.degree(node)}


class ShortestPathTask(BaseTask):
    def generate_stimuli(self, G: nx.Graph, num_stimuli: int, seed: Optional[int] = None) -> List[TaskParameters]:
        """Generate random pairs of nodes that have a valid path between them, avoiding duplicates"""
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        stimuli = []
        nodes = list(G.nodes())

        used_paths = set()

        if len(nodes) < 2:
            print("Warning: Graph has fewer than 2 nodes")
            return []

        max_attempts = len(nodes) * 10  # Set this as a rule of thumb
        attempts = 0

        while len(stimuli) < num_stimuli and attempts < max_attempts:
            try:
                start, end = random.sample(nodes, 2)
                path_key = tuple(sorted([start, end]))

                if path_key not in used_paths and nx.has_path(G, start, end):
                    print(f"Found valid path between {start} and {end}")
                    used_paths.add(path_key)
                    stimuli.append(
                        TaskParameters.create_shortest_path_params(start, end))

                attempts += 1

            except ValueError as e:
                print(f"Error during sampling: {e}")
                break

        if not stimuli:
            print(
                f"Warning: Could not generate any valid node pairs after {attempts} attempts")
            print("Graph connectivity info:")
            print(f"Number of nodes: {G.number_of_nodes()}")
            print(f"Number of edges: {G.number_of_edges()}")
            print(f"Is connected: {nx.is_connected(G)}")

        return stimuli

    def compute_ground_truth(self, G: nx.Graph, parameters: TaskParameters) -> Dict:
        start = parameters.params["start_node"]
        end = parameters.params["end_node"]
        try:
            path = nx.shortest_path(G, start, end)
            return {"path": path, "length": len(path) - 1}
        except nx.NetworkXNoPath:
            print(f"No path exists from {start} to {end}")
            return {"path": None, "length": None}


class CommonNeighboursTask(BaseTask):
    def generate_stimuli(self, G: nx.Graph, num_stimuli: int, seed: Optional[int] = None) -> List[TaskParameters]:
        """
        Generate random pairs of nodes that share at least one common neighbor.
        Attempts to find pairs with different numbers of common neighbors for varying difficulty.
        Ensures no duplicate pairs (including reverse order).
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        stimuli = []
        nodes = list(G.nodes())

        if len(nodes) < 2:
            print("Warning: Graph has fewer than 2 nodes")
            return []

        used_pairs = set()

        node_pairs_info = {}
        for node in nodes:
            node_neighbors = set(G.neighbors(node))

            for other_node in nodes:
                if other_node <= node:
                    continue

                other_neighbors = set(G.neighbors(other_node))

                common = node_neighbors & other_neighbors

                if common:  # Only store if they have common neighbors
                    node_pairs_info[(node, other_node)] = len(common)

        # Convert to list of valid pairs
        valid_pairs = list(node_pairs_info.keys())

        if not valid_pairs:
            print("Warning: No valid node pairs found with common neighbors")
            return []

        max_attempts = len(valid_pairs) * 2
        attempts = 0

        if num_stimuli > 1:
            harder_pairs = [pair for pair,
                            count in node_pairs_info.items() if count > 1]
            if harder_pairs:
                pair = random.choice(harder_pairs)
                stimuli.append(
                    TaskParameters.create_common_neighbours_params(pair[0], pair[1]))
                used_pairs.add(pair)

        while len(stimuli) < num_stimuli and attempts < max_attempts:
            if not valid_pairs:
                break

            remaining_pairs = [p for p in valid_pairs if p not in used_pairs]
            if not remaining_pairs:
                break

            pair = random.choice(remaining_pairs)
            used_pairs.add(pair)
            stimuli.append(
                TaskParameters.create_common_neighbours_params(pair[0], pair[1]))

            attempts += 1

        for stimulus in stimuli:
            start = stimulus.params["start_node"]
            end = stimulus.params["end_node"]
            print(
                f"Selected nodes {start} and {end} with common neighbors: {node_pairs_info[(start, end)]}")

        if not stimuli:
            print(
                f"Warning: Could not generate any valid node pairs after {attempts} attempts")
            print("Graph connectivity info:")
            print(f"Number of nodes: {G.number_of_nodes()}")
            print(f"Number of edges: {G.number_of_edges()}")

        return stimuli

    def compute_ground_truth(self, G: nx.Graph, parameters: TaskParameters) -> Dict:
        """
        Compute the ground truth for a pair of nodes: common neighbors and their count.
        """
        node1 = parameters.params["start_node"]
        node2 = parameters.params["end_node"]

        common_neighbors = list(set(G.neighbors(node1))
                                & set(G.neighbors(node2)))
        return {
            "neighbours": common_neighbors,
            "number": len(common_neighbors)
        }


class MaxCliqueTask(BaseTask):
    def generate_stimuli(self, G: nx.Graph, num_stimuli: int, seed: Optional[int] = None) -> List[TaskParameters]:
        # No randomization needed for this task
        return [TaskParameters(type="graph", params={})]

    def compute_ground_truth(self, graph_name: str, parameters: TaskParameters) -> Dict:

        try:
            clique_part = graph_name.split('_')[-1]
            if not clique_part.startswith('c'):
                raise ValueError(
                    f"Graph name does not contain 'c' part: {graph_name}")
            clique_size = int(clique_part.replace('c', ''))
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Failed to extract vertex cover size from graph name '{graph_name}': {e}")

        return {"clique": {}, "size": clique_size}


class MinVertexCoverTask(BaseTask):
    def generate_stimuli(self, G: nx.Graph, num_stimuli: int, seed: Optional[int] = None) -> List[TaskParameters]:
        # No randomization needed for this task
        return [TaskParameters(type="graph", params={})]

    def compute_ground_truth(self, graph_name: str, parameters: TaskParameters) -> Dict:
        # The graph name is expected to be in the format "graph_N_vcX.lst" for this task to work, simplify the computation
        try:
            vc_part = graph_name.split('_')[-1]
            if not vc_part.startswith('vc'):
                raise ValueError(
                    f"Graph name does not contain 'vc' part: {graph_name}")
            vc_size = int(vc_part.replace('vc', ''))
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Failed to extract vertex cover size from graph name '{graph_name}': {e}")

        print(f'vc_size: {vc_size}')
        return {"cover": {}, "size": vc_size}


class AdjacencyListTask(BaseTask):
    def generate_stimuli(self, G: nx.Graph, num_stimuli: int, seed: Optional[int] = None) -> List[TaskParameters]:
        """
        For adjacency list task, we only need one stimulus per graph since we're 
        displaying the full adjacency list
        """
        # No randomization needed for this task
        return [TaskParameters(type="graph", params={})]

    def compute_ground_truth(self, G: nx.Graph, parameters: TaskParameters) -> Dict:
        """
        Read the adjacency list from the .lst file and format it to match the expected output format
        Returns a dictionary containing the formatted adjacency list
        Format: "<Vertex-ID>: comma-separated-neighbors; ..."
        """
        try:
            # Get the file path directly from the graph_file attribute
            file_path = getattr(G, 'graph_file', None)
            if not file_path:
                raise ValueError("Graph file path not found in graph object")

            if not Path(file_path).is_file():
                raise FileNotFoundError(f"Graph file not found: {file_path}")

            # Read and format the adjacency list
            formatted_list = read_adjacency_list(file_path)

            return {
                "adjacency_list": formatted_list
            }
        except Exception as e:
            print(f"Error in compute_ground_truth: {str(e)}")
            print(f"Graph file path: {getattr(G, 'graph_file', 'Not found')}")
            raise


class TaskFactory:
    """Factory for creating task instances"""
    TASKS = {
        "vertex_degree": VertexDegreeTask,
        "shortest_path": ShortestPathTask,
        "max_clique": MaxCliqueTask,
        "min_vertex_cover": MinVertexCoverTask,
        "common_neighbours": CommonNeighboursTask,
        "adjacency_list": AdjacencyListTask
    }

    @classmethod
    def create_task(cls, task_name: str, graph_type: str):
        task_class = cls.TASKS.get(task_name)
        if not task_class:
            raise ValueError(f"Unknown task: {task_name}")
        return task_class(graph_type)


def load_graph(file_path: Path, graph_type: str) -> nx.Graph:
    """Load graph from file with proper directed/undirected handling"""
    # Create appropriate graph type
    G = nx.DiGraph() if graph_type == "di" else nx.Graph()

    # Store the absolute file path in the graph object
    G.graph_file = str(file_path.absolute())

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.replace(':', ' ').split()
            if not parts:
                continue

            node = parts[0]
            neighbors = parts[1:]

            if neighbors:
                G.add_edges_from((node, neighbor) for neighbor in neighbors)

    return G
