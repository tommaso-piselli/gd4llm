import base64
import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional

# ! GENERAL FUNCTIONS


def encode_image(image_path: Path) -> str:
    """Convert an image file to base64 string for API transmission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_config(config_path: Path) -> Dict:
    """Load JSON configuration file with error handling"""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")

# !ADJACENCY LIST TASK


def read_adjacency_list(file_path: Path) -> str:
    """
    Read an adjacency list from a .lst file and format it into the required string format.
    This is used for the adjacency list task.
    Returns: string in format "<Vertex-ID>: comma-separated-neighbors; ..."
    """
    adjacency_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(':')
            if len(parts) != 2:
                continue

            node = parts[0].strip()
            neighbors = [n for n in parts[1].strip().split() if n]

            adjacency_dict[node] = neighbors

    # Format the adjacency list in the required format
    formatted_entries = []
    for node in sorted(adjacency_dict.keys(), key=lambda x: int(x)):
        neighbors = adjacency_dict[node]
        sorted_neighbors = sorted(neighbors, key=lambda x: int(x))
        formatted_entry = f"{node}: {','.join(sorted_neighbors)}"
        formatted_entries.append(formatted_entry)

    return "; ".join(formatted_entries)


def parse_adjacency_list(adj_list_str: str) -> Dict[str, set]:
    """
    Parse adjacency list string into a dictionary
    Format expected: "1: 9; 2: 10; 3: 10" etc.
    Where each entry is node: comma-separated-neighbors
    Returns: Dictionary with nodes as keys and sets of neighbors as values
    """
    result = {}
    # Split the string into individual node entries by semicolon
    entries = [e.strip() for e in adj_list_str.split(';') if e.strip()]

    for entry in entries:
        if ':' in entry:
            node, neighbors = entry.split(':')
            node = node.strip()
            # Split neighbors by comma
            neighbors = set(n.strip()
                            for n in neighbors.split(',') if n.strip())
            result[node] = neighbors
    return result


def parse_al(response: str) -> Optional[Dict]:
    """
    Parse an adjacency list from LLM response. This is used not for the other tasks.
    Returns a dictionary with an 'adjacency_list' key in the format "1: 2,3; 4: 5,6".
    """

    cleaned_response = re.sub(
        r'^.*?[Aa]djacency\s+[Ll]ist\s*:', '', response, 1).strip()

    # If no adjacency list keyword found, try using the whole response
    if cleaned_response == response:
        cleaned_response = response

    entries = []
    raw_entries = [entry.strip()
                   for entry in cleaned_response.split(';') if entry.strip()]

    patterns = [
        r'(\d+)\s*:\s*\[(.*?)\]',
        r'(\d+)\s*:\s*(?:\[?([\d\s,]+)\]?)',
        r'\[?(\d+)\]?\s*:\s*(?:\[?([\d\s,]+)\]?)',
        r'[Nn]ode\s*(\d+)\s*:\s*(?:\[?([\d\s,]+)\]?)',
    ]

    for entry in raw_entries:
        entry_match = None
        for pattern in patterns:
            match = re.search(pattern, entry)
            if match:
                entry_match = match
                break

        if entry_match:
            node = entry_match.group(1).strip()

            if len(entry_match.groups()) > 1 and entry_match.group(2) is not None:
                neighbors_str = entry_match.group(2).strip()
            else:
                neighbors_str = ""

            neighbors = []
            if neighbors_str:

                if "node" in neighbors_str.lower():
                    neighbors_str = neighbors_str.lower().split("node")[
                        0].strip()

                neighbors = clean_number_list(neighbors_str)

            neighbors_str = ','.join(str(n) for n in neighbors)
            entries.append(f"{node}: {neighbors_str}")

    if not entries:
        return None

    try:
        entries.sort(key=lambda x: int(x.split(':')[0]))
    except (ValueError, IndexError):
        pass

    result = "; ".join(entries)

    return {"adjacency_list": result}


def compute_jaccard_similarity_adjlist(list1_str: str, list2_str: str) -> float:
    """
    Compute Jaccard similarity between two adjacency lists represented as strings
    Format expected: "1: 9; 2: 10; 3: 10" etc.
    Where each entry is node: comma-separated-neighbors
    """

    dict1 = parse_adjacency_list(list1_str)
    dict2 = parse_adjacency_list(list2_str)

    all_nodes = set(dict1.keys()) | set(dict2.keys())

    total_intersection = 0
    total_union = 0

    for node in all_nodes:
        neighbors1 = dict1.get(node, set())
        neighbors2 = dict2.get(node, set())

        total_intersection += len(neighbors1 & neighbors2)
        total_union += len(neighbors1 | neighbors2)

    if total_union == 0:
        return 1.0

    return round(total_intersection / total_union, 3)


def compute_average_node_jaccard_similarity_adjlist(list1_str: str, list2_str: str) -> float:
    """
    Compute Jaccard similarity between two adjacency lists by:
    1. Computing Jaccard similarity for each node's neighborhood
    2. Taking the average across all nodes

    Format expected: "1: 9; 2: 10; 3: 10" etc.
    Where each entry is node: comma-separated-neighbors

    Returns: Average Jaccard similarity across all nodes
    """

    dict1 = parse_adjacency_list(list1_str)
    dict2 = parse_adjacency_list(list2_str)

    all_nodes = set(dict1.keys()) | set(dict2.keys())

    if not all_nodes:
        return 1.0

    node_similarities = []
    for node in all_nodes:

        neighbors1 = dict1.get(node, set())
        neighbors2 = dict2.get(node, set())

        intersection_size = len(neighbors1 & neighbors2)
        union_size = len(neighbors1 | neighbors2)

        if union_size == 0:

            node_similarities.append(1.0)
        else:
            node_similarities.append(intersection_size / union_size)

    average_similarity = sum(node_similarities) / len(node_similarities)

    return round(average_similarity, 3)

# !COMMON NEIGHBORS TASK


def parse_common_neighbors(response: str) -> Optional[List[str]]:
    """
    Extract common neighbors from LLM response.
    Returns a list of node IDs or an empty list if no neighbors are found.
    """
    # Case 1: Check for explicit "no common neighbors" or "0 common neighbors" statements
    no_neighbors_patterns = [
        r"[Nn]o\s+common\s+neighbou?rs?",
        r"[Tt]here\s+are\s+no\s+common\s+neighbou?rs?",
        r"[Nn]umber\s+of\s+common\s+neighbou?rs?\s+is\s+0",
        r"[Tt]he\s+number\s+of\s+common\s+neighbou?rs?\s+is\s+0",
        r"[Zz]ero\s+common\s+neighbou?rs?",
        r"[Nn]one?(?:\s+of\s+the\s+vertices|\s+of\s+the\s+nodes)?\s+are\s+common\s+neighbou?rs?",
        r"[Nn]eighbou?rs:?\s*\[\s*\]\s*,?\s*[Nn]umber:?\s*0"
    ]

    for pattern in no_neighbors_patterns:
        if re.search(pattern, response):
            return []  # Return empty list instead of None

    # Case 2: Look for explicit empty neighbors representation
    empty_neighbors_patterns = [
        r"[Nn]eighbou?rs:?\s*\[\s*\]",
        r"[Nn]eighbou?rs:?\s*\{\s*\}",
        r"[Nn]eighbou?rs:?\s*None",
        r"[Nn]eighbou?rs:?\s*\(\s*\)",
        r"\*\*[Nn]eighbou?rs\*\*:?\s*\[\s*\]"
    ]

    for pattern in empty_neighbors_patterns:
        if re.search(pattern, response):
            return []  # Return empty list instead of None

    # Case 3: Parse neighbors normally
    neighbors_patterns = [
        r"[Nn]eighbou?rs:?\s*\[(.*?)\]",
        r"[Nn]eighbou?rs:?\s*`(.*?)`",
        r"\*\*[Nn]eighbou?rs?\*\*:?\s*\[(.*?)\]",
        r"\*\*[Nn]eighbou?rs?\*\*:?\s*`(.*?)`",
        r"[Cc]ommon\s+[Nn]eighbou?rs?:?\s*\[(.*?)\]",
        r"[Cc]ommon\s+[Nn]eighbou?rs?:?\s*`(.*?)`",
        r"[Nn]eighbou?rs:?\s*([\d\s,]+)(?=,\s*[Nn]umber:)",
        r"[Nn]eighbou?rs:?\s*([\d\s,\*]+)(?=\s+[Nn]umber:|\s*$|\s|,|\.|;)",
        r"[Nn]eighbou?rs:?\s*(?:`|')([\d\s,\*]+)(?:`|')(?=\s*[Nn]umber:|$|\s|,|\.|;)",
        r"[Nn]eighbou?rs:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Nn]umber)",
        r"[Cc]ommon\s+[Nn]eighbou?rs?:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Nn]umber)"
    ]

    for pattern in neighbors_patterns:
        match = re.search(pattern, response)
        if match and match.group(1).strip():
            neighbors_str = match.group(1).strip()

            # Remove "Number:" and anything after if captured
            number_match = re.search(r"(.*?)(?:,\s*[Nn]umber:)", neighbors_str)
            if number_match:
                neighbors_str = number_match.group(1).strip()
            elif "number" in neighbors_str.lower():
                neighbors_str = neighbors_str.lower().split("number")[
                    0].strip()

            # Remove "Size:" and anything after if captured
            if "size" in neighbors_str.lower():
                neighbors_str = neighbors_str.lower().split("size")[0].strip()

            return clean_number_list(neighbors_str)

    return []


def parse_common_neighbors_count(response: str, neighbors: Optional[List[str]] = None) -> Optional[int]:
    """
    Extract the count of common neighbors from LLM response.
    If neighbors list is provided, uses its length as a fallback if no explicit count is found.
    Returns the count as an integer or 0 if no common neighbors.
    """
    # Case 1: Check for explicit "no common neighbors" or "0 common neighbors" statements
    no_neighbors_patterns = [
        r"[Nn]o\s+common\s+neighbou?rs?",
        r"[Tt]here\s+are\s+no\s+common\s+neighbou?rs?",
        r"[Nn]umber\s+of\s+common\s+neighbou?rs?\s+is\s+0",
        r"[Tt]he\s+number\s+of\s+common\s+neighbou?rs?\s+is\s+0",
        r"[Zz]ero\s+common\s+neighbou?rs?",
        r"[Nn]one?(?:\s+of\s+the\s+vertices|\s+of\s+the\s+nodes)?\s+are\s+common\s+neighbou?rs?",
        r"[Nn]umber:?\s*0"
    ]

    for pattern in no_neighbors_patterns:
        if re.search(pattern, response):
            return 0  # Return 0 instead of None for no neighbors

    # Case 2: Look for explicit number
    number_patterns = [
        r"[Nn]umber:?\s*(\d+)",
        r"[Nn]umber\s+of\s+[Nn]eighbou?rs?:?\s*(\d+)",
        r"\*\*[Nn]umber\*\*:?\s*(\d+)",
        r"[Tt]here\s+(?:are|is)\s+(\d+)\s+[Nn]eighbou?rs?",
        r"[Ss]ize:?\s*(\d+)",
        r"[Tt]otal:?\s*(\d+)",
        r"[Cc]ommon\s+[Nn]eighbou?rs?:?\s*(\d+)",
        r"[Tt]he\s+nodes?\s+(?:has|have)\s+(\d+)\s+[Nn]eighbou?rs?",
        r"[Tt]he\s+number\s+of\s+common\s+neighbou?rs?\s+is\s+(\d+)"
    ]

    for pattern in number_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                count = int(match.group(1))
                return count  # Return the count, even if 0
            except ValueError:
                continue

    # Case 3: Fallback to neighbors length if provided
    if neighbors is not None:
        return len(neighbors)

    # If no count was found and neighbors is None, return 0 (best guess for "no neighbors")
    return 0


def clean_neighbors_string(neighbors_str: str) -> str:
    """
    Clean and standardize the neighbors string by:
    - Removing square brackets
    - Handling both comma and space separated values
    - Removing any extra whitespace or empty strings
    """
    # Remove square brackets if present
    neighbors_str = neighbors_str.strip('[]')

    # Split by either commas or spaces
    if ',' in neighbors_str:
        neighbors = [n.strip() for n in neighbors_str.split(',')]
    else:
        neighbors = [n.strip() for n in neighbors_str.split()]

    # Filter out empty strings and sort numerically
    neighbors = [n for n in neighbors if n]
    neighbors.sort(key=lambda x: int(x))

    return ','.join(neighbors)


def compute_average_node_jaccard_similarity_cn(predicted: Dict, ground_truth: Dict) -> float:
    """
    Compute average Jaccard similarity for common neighbors,
    similar to how it's done for adjacency lists.
    """
    if not predicted or not ground_truth:
        return 0.0

    pred_neighbors = predicted.get("neighbours", [])
    gt_neighbors = ground_truth.get("neighbours", [])

    # Handle None values
    if pred_neighbors is None or gt_neighbors is None:
        return 0.0

    # Convert all elements to strings for consistent comparison
    pred_set = {str(x) for x in pred_neighbors}
    gt_set = {str(x) for x in gt_neighbors}

    # If both sets are empty, consider it a perfect match
    if not pred_set and not gt_set:
        return 1.0

    # Calculate individual Jaccard similarities
    all_nodes = pred_set | gt_set
    node_similarities = []

    for node in sorted(all_nodes):  # Sort for consistency
        # For each node, compare its presence/absence in both sets
        pred_has = node in pred_set
        gt_has = node in gt_set

        if not pred_has and not gt_has:
            node_similarities.append(1.0)  # Perfect match (both don't have it)
        elif pred_has and gt_has:
            node_similarities.append(1.0)  # Perfect match (both have it)
        else:
            node_similarities.append(0.0)  # Mismatch

    # If no nodes to compare, return perfect match
    if not node_similarities:
        return 1.0

    # Calculate average similarity
    average_similarity = sum(node_similarities) / len(node_similarities)
    return round(average_similarity, 3)

# ! SHORTEST PATH TASK


def check_edge_exists(node1: str, node2: str, adjacency_dict: Dict[str, List[str]]) -> bool:
    """
    Check if an edge exists between two nodes in the graph.
    For undirected graphs, check both directions.
    """
    # Convert to strings to ensure consistent comparison
    node1, node2 = str(node1), str(node2)

    # Get neighbors of both nodes
    neighbors1 = set(str(n) for n in adjacency_dict.get(node1, []))
    neighbors2 = set(str(n) for n in adjacency_dict.get(node2, []))

    # Check if either node is a neighbor of the other
    return node2 in neighbors1 or node1 in neighbors2


def count_actual_edges(path: List[str], adjacency_dict: Dict[str, List[str]]) -> int:
    """
    Count how many edges in the path actually exist in the graph.
    """
    if not path or len(path) < 2:
        return 0

    actual_edges = 0
    for i in range(len(path) - 1):
        if check_edge_exists(path[i], path[i + 1], adjacency_dict):
            actual_edges += 1

    return actual_edges


def compute_shortest_path_accuracy(predicted: Dict, ground_truth: Dict, lst_file: str) -> float:
    """
    Compute accuracy for shortest path task using the formula:
    min(gt_len/pred_len, pred_len/gt_len) * actual_edges/pred_len

    Args:
        predicted: Dictionary with 'length' and 'path' keys
        ground_truth: Dictionary with 'length' and 'path' keys
        lst_file: Path to the .lst file containing the graph structure
    """
    # Handle invalid inputs
    if not predicted or not ground_truth:
        return 0.0

    pred_length = predicted.get('length')
    gt_length = ground_truth.get('length')
    pred_path = predicted.get('path')

    # If any required value is missing, return 0
    if pred_length is None or gt_length is None or not pred_path:
        return 0.0

    try:
        # Read adjacency list from file
        adjacency_dict = {}
        with open(lst_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        node = parts[0].strip()
                        neighbors = [n.strip()
                                     for n in parts[1].strip().split()]
                        adjacency_dict[node] = neighbors

        # Count actual edges in predicted path
        actual_edges = count_actual_edges(pred_path, adjacency_dict)

        # Compute length ratio part
        ratio = min(gt_length/pred_length, pred_length/gt_length)

        # Compute edge accuracy part
        edge_accuracy = actual_edges/pred_length if pred_length > 0 else 0.0

        # Compute final accuracy
        accuracy = ratio * edge_accuracy

        return round(accuracy, 3)

    except Exception as e:
        print(f"Error computing shortest path accuracy: {e}")
        return 0.0


def parse_path(response: str) -> Optional[List[str]]:
    """
    Extract path from the response using various patterns.
    Returns a list of node IDs or None if no path is found.
    """
    # Path patterns from most specific to most general
    path_patterns = [
        # Bold markdown with brackets and colon
        r"\*\*Path:\*\*\s*\[(.*?)\]",

        # Bold markdown without colon
        r"\*\*Path\*\*:?\s*\[(.*?)\]",

        # List item with bold markdown
        r"-\s*\*\*Path:\*\*\s*\[(.*?)\]",

        # Backtick format
        r"`Path:\s*([\d\s,]+)",

        # Path inside brackets
        r"[Pp]ath:?\s*\[(.*?)\]",

        # Path without brackets but followed by Length
        r"[Pp]ath:?\s*([\d\s,]+)(?=[Ll]ength|$)",

        # Arrow notation after "The path is:"
        r"[Tt]he path is:?\s*((?:\d+\s*→\s*)+\d+)",

        # Arrow notation in descriptive text
        r"path (?:is|found):?\s*((?:\d+\s*(?:→|to)\s*)+\d+)",

        # Path described with "to" format
        r"(\d+(?:\s+to\s+\d+)+)",

        # Shortest path format
        r"[Ss]hortest\s+[Pp]ath:?\s*((?:\d+(?:\s*[,]\s*)?)+)",

        # Generic path with any separator
        r"[Pp]ath:?\s*((?:\d+(?:\s*[,\*]\s*)*)+)(?=\s|$|,\s*[Ll]ength)"
    ]

    # Try all patterns
    for pattern in path_patterns:
        match = re.search(pattern, response)
        if match:
            path_str = match.group(1).strip()
            # If it contains "length", truncate at that point
            if "length" in path_str.lower():
                path_str = path_str.lower().split("length")[0].strip()
            return clean_number_list(path_str)

    # Look for alternative path formats in the response
    # Check for step 3 where the path is often mentioned
    step3_pattern = r"[Ss]tep\s*3:.*?path\s*(?:\d+\s*→\s*)+\d+"
    match = re.search(step3_pattern, response, re.DOTALL)
    if match:
        # Extract the path from this section
        path_text = match.group(0)
        arrow_pattern = r"(\d+\s*→\s*\d+(?:\s*→\s*\d+)*)"
        arrow_match = re.search(arrow_pattern, path_text)
        if arrow_match:
            return clean_number_list(arrow_match.group(1))

    # If still nothing found, check for verification sections
    verification_pattern = r"[Vv]erification:.*?[Nn]ode\s*\d+.*?[Nn]ode\s*\d+"
    match = re.search(verification_pattern, response, re.DOTALL)
    if match:
        verification_text = match.group(0)
        # Look for node connections
        node_connections = re.findall(
            r"[Nn]ode\s*(\d+)\s*is connected to\s*[Nn]ode\s*(\d+)", verification_text)
        if node_connections:
            # Reconstruct the path from verified connections
            path = [node_connections[0][0]]
            for conn in node_connections:
                path.append(conn[1])
            return path

    # If still nothing found, look for any sequences that might be paths
    sequence_pattern = r"(?:^|\s|:)(\d+(?:\s*(?:,|→|to)\s*\d+){1,10})(?:\s|$|,|\.|;)"
    matches = re.finditer(sequence_pattern, response)
    for match in matches:
        seq = match.group(1)
        if "," in seq or "→" in seq or "to" in seq:
            return clean_number_list(seq)

    return None


def parse_length(response: str, parsed_path: Optional[List[str]] = None) -> Optional[int]:
    """
    Extract length from the response using various patterns.
    Returns the length as an integer or None if no length is found.

    Args:
        response: The LLM response text
        parsed_path: Optional list of path nodes to validate against the parsed length
    """
    # First look for final statements about length with priority
    final_statement_patterns = [
        # Path: [2, 10, 3, 1], Length: 3 at end
        r"[Pp]ath:.*?,[Ll]ength:\s*(\d+)\s*$",
        # Length: 3 at end of response
        r"[Ll]ength:\s*(\d+)\s*$",
        # Length is 3 edges at end
        r"[Ll]ength\s*(?:is|=|:)?\s*(\d+)\s*(?:edges?)?\s*$",
    ]

    for pattern in final_statement_patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            try:
                length = int(match.group(1).strip())
                # If we have a path, validate the length makes sense
                if parsed_path and len(parsed_path) > 1:
                    expected_length = len(parsed_path) - 1
                    # If the parsed length matches the path length or is within a small margin
                    if length == expected_length or abs(length - expected_length) <= 1:
                        return length
                else:
                    return length
            except ValueError:
                continue

    # Specific patterns with more context to avoid mistaken matches
    contextual_patterns = [
        # Explicit length statements
        # Path... Length: 3
        r"[Pp]ath.*?[Ll]ength:?\s*(\d+)(?:\s*edges?)?",
        # shortest path... Length: 3
        r"[Ss]hortest\s+path.*?[Ll]ength:?\s*(\d+)(?:\s*edges?)?",
        # The length of the path is 3
        r"[Tt]he\s+length\s+of\s+the\s+(?:shortest\s+)?path\s+is\s+(\d+)",
        # Path length: 3
        r"[Pp]ath\s+length:?\s*(\d+)(?!\s*\d)",
        # The path has 3 edges
        r"[Tt]he\s+path\s+(?:has|contains)\s+(\d+)\s+edges",
        # Length of 3 steps/edges
        r"[Ll]ength\s+of\s+(\d+)\s+(?:steps|edges)",
        # Path with 3 edges
        r"[Pp]ath\s+with\s+(\d+)\s+edges",
        # Path of length 3
        r"[Pp]ath\s+of\s+length\s+(\d+)",
        # Path consists of 3 edges
        r"[Pp]ath\s+consists\s+of\s+(\d+)\s+edges",
    ]

    for pattern in contextual_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                length = int(match.group(1).strip())
                # If we have a path, validate the length
                if parsed_path and len(parsed_path) > 1:
                    expected_length = len(parsed_path) - 1
                    # If the parsed length is dramatically different, it might be wrong
                    if abs(length - expected_length) > 2:
                        continue
                return length
            except ValueError:
                continue

    # Original patterns as fallback but with safeguards
    length_patterns = [
        # Bold markdown with colon - ensure we're not matching node numbers
        r"\*\*Length:\*\*\s*(\d+)(?!\s*\d)",
        r"\*\*Length\*\*:?\s*(\d+)(?!\s*\d)",
        r"-\s*\*\*Length:\*\*\s*(\d+)(?!\s*\d)",

        # Standard formats with negative lookahead to avoid node numbers
        r"[Ll]ength:?\s*(\d+)(?!\s*[,\d])",

        # Length mentioned in text with safeguards
        r"[Tt]he path length is (\d+)(?!\s*\d)",
        r"path.*?length.*?(\d+)\s*edges?"
    ]

    for pattern in length_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                length = int(match.group(1).strip())

                # Validate the parsed length against the path
                if parsed_path and len(parsed_path) > 1:
                    expected_length = len(parsed_path) - 1
                    # If there's a huge mismatch, prefer the path-based length
                    if abs(length - expected_length) > 2:
                        continue

                return length
            except ValueError:
                continue

    # Fallback: If we have a path, use its length
    if parsed_path and len(parsed_path) > 1:
        return len(parsed_path) - 1

    return None


def clean_number_list(number_str: str) -> List[str]:
    """
    Clean and convert a string of numbers into a list.
    Handles various formats including asterisks, commas, spaces, and brackets.
    """
    # Remove brackets and other unwanted characters
    for char in "[]{}*()":
        number_str = number_str.replace(char, '')

    # Replace arrow notation and "to" with commas
    number_str = number_str.replace('→', ',').replace(' to ', ',')

    # Split by comma or space and clean
    numbers = []
    for item in re.split(r'[,\s]+', number_str):
        item = item.strip()
        if item and item.isdigit():
            numbers.append(item)

    return numbers


def parse_shortest_path(response: str) -> Dict:
    """
    Extract shortest path information from LLM response with improved accuracy.
    Returns a dictionary with 'path' and 'length' keys.
    """
    # Parse path first
    path = parse_path(response)

    # Parse length with path validation
    length = parse_length(response, path)

    # If we have a path but no length, calculate it from the path
    if path and not length and len(path) > 1:
        length = len(path) - 1

    # For logging/debugging purposes
    # print(f"Parsed path: {path}, length: {length}")

    return {
        "path": path,
        "length": length
    }

# ! VEREX COVER TASK


def get_total_edges(lst_file: str) -> int:
    """
    Get the total number of edges in the graph from the .lst file
    """
    edges = set()
    with open(lst_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(':')
            if len(parts) != 2:
                continue

            source = parts[0].strip()
            targets = parts[1].strip().split()

            # Add edges (maintaining undirected graph properties)
            for target in targets:
                # Sort vertices to ensure consistent edge representation
                edge = tuple(sorted([source, target]))
                edges.add(edge)

    return len(edges)


def count_uncovered_edges(vertex_cover: List[str], lst_file: str) -> int:
    """
    Count the number of edges that remain after removing the vertices in the vertex cover
    and their incident edges.

    Args:
        vertex_cover: List of vertex IDs in the cover
        lst_file: Path to the .lst file containing the graph structure

    Returns:
        Number of edges that are not covered by the vertex cover
    """
    # Convert vertex cover to set for O(1) lookup
    cover_set = set(str(v) for v in vertex_cover)

    # Keep track of remaining edges
    uncovered_edges = set()

    with open(lst_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(':')
            if len(parts) != 2:
                continue

            source = parts[0].strip()
            # Skip if this vertex is in the cover
            if source in cover_set:
                continue

            targets = parts[1].strip().split()

            # For each edge where neither endpoint is in the cover, count it
            for target in targets:
                if target not in cover_set:
                    # Sort vertices to ensure consistent edge representation
                    edge = tuple(sorted([source, target]))
                    uncovered_edges.add(edge)

    return len(uncovered_edges)


def compute_vertex_cover_accuracy(predicted: Dict, ground_truth: Dict, lst_file: str) -> float:
    """
    Compute accuracy for vertex cover task using the formula:
    min(gt_size/pred_size, pred_size/gt_size) * \
        (1 - uncovered_edges/total_edges)

    Args:
        predicted: Dictionary with 'cover' and 'size' keys
        ground_truth: Dictionary with 'cover' and 'size' keys
        lst_file: Path to the .lst file containing the graph structure
    """
    # Handle invalid inputs
    if not predicted or not ground_truth:
        return 0.0

    pred_size = predicted.get('size')
    gt_size = ground_truth.get('size')
    pred_cover = predicted.get('cover')

    # If any required value is missing, return 0
    if pred_size is None or gt_size is None or not pred_cover:
        return 0.0

    try:
        # Compute cardinality ratio
        ratio = min(gt_size/pred_size, pred_size/gt_size)

        # Get total edges in the graph
        total_edges = get_total_edges(lst_file)
        if total_edges == 0:
            return 1.0 if pred_size == gt_size else 0.0

        # Count uncovered edges
        uncovered_edges = count_uncovered_edges(pred_cover, lst_file)

        # Compute coverage accuracy
        coverage_accuracy = 1 - (uncovered_edges / total_edges)

        # Compute final accuracy
        accuracy = ratio * coverage_accuracy

        return round(accuracy, 3)

    except Exception as e:
        print(f"Error computing vertex cover accuracy: {e}")
        return 0.0


def parse_vertex_cover_nodes(response: str) -> Optional[List[str]]:
    """
    Extract the nodes of the minimum vertex cover from LLM response.
    Returns a list of node IDs or None if no cover is found.
    """
    # First check for multi-line formats with "Cover:" on its own line
    cover_line_pattern = r"(?:After checking,|Following this approach,|After careful analysis|I've determined).*?\n[Cc]over:?\s*([\d\s,]+)"
    match = re.search(cover_line_pattern, response, re.DOTALL)
    if match and match.group(1).strip():
        cover_str = match.group(1).strip()
        # Remove "Size:" and anything after if captured
        if "size" in cover_str.lower():
            cover_str = cover_str.lower().split("size")[0].strip()
        return clean_number_list(cover_str)

    # Then try regular patterns
    cover_patterns = [
        r"[Cc]over:?\s*\{(.*?)\}",
        r"[Cc]over:?\s*\[(.*?)\]",
        r"\*\*[Cc]over:\*\*\s*\[(.*?)\]",
        r"\*\*[Cc]over\*\*:?\s*\{(.*?)\}",
        r"\*\*[Cc]over\*\*:?\s*\[(.*?)\]",
        r"\*\*[Cc]over:\*\*\s*([\d\s,]+)",  # Pattern for: **Cover:** 1, 2, 3
        r"[Vv]ertex\s+[Cc]over:?\s*\{(.*?)\}",
        r"[Vv]ertex\s+[Cc]over:?\s*\[(.*?)\]",
        r"[Mm]inimum\s+[Vv]ertex\s+[Cc]over:?\s*\[(.*?)\]",
        r"[Mm]inimum\s+[Cc]over:?\s*\[(.*?)\]",
        # Specific patterns for list formats
        r"-\s*(?:\*\*)?[Cc]over:?(?:\*\*)?\s*\[(.*?)\]",
        # Pattern for: - Cover: 3, 4, 6, 7, 8, 13
        r"-\s*(?:\*\*)?[Cc]over:?(?:\*\*)?\s*([\d\s,]+)",
        # Specific pattern for format with comma-size
        r"[Cc]over:?\s*([\d\s,]+)(?=,\s*[Ss]ize:)",
        # Standalone Cover: with numbers pattern
        # Matches "Cover: 1, 2, 3" on its own line
        r"^[Cc]over:?\s*([\d\s,]+)$",
        # More specific patterns with boundaries
        r"[Cc]over:?\s*([\d\s,\*]+)(?=\s+[Ss]ize:|\s*$|\s|,|\.|;)",
        r"[Vv]ertex\s+[Cc]over:?\s*([\d\s,\*]+)(?=\s*[Ss]ize:|$|\s|,|\.|;)",
        r"[Mm]inimum\s+[Cc]over:?\s*([\d\s,\*]+)(?=\s*[Ss]ize:|$|\s|,|\.|;)",
        # Generic patterns
        r"[Cc]over:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Ss]ize)",
        r"[Vv]ertex\s+[Cc]over:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Ss]ize)",
        r"[Mm]inimum\s+[Cc]over:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Ss]ize)"
    ]

    for pattern in cover_patterns:
        match = re.search(pattern, response)
        if match:
            cover_str = match.group(1).strip()

            # Remove "Size:" and anything after if captured
            size_match = re.search(r"(.*?)(?:,\s*[Ss]ize:)", cover_str)
            if size_match:
                cover_str = size_match.group(1).strip()
            elif "size" in cover_str.lower():
                cover_str = cover_str.lower().split("size")[0].strip()

            return clean_number_list(cover_str)

    return None


def parse_vertex_cover_size(response: str, cover: Optional[List[str]] = None) -> Optional[int]:
    """
    Extract the size of the minimum vertex cover from LLM response.
    If cover list is provided, uses its length as a fallback if no explicit size is found.
    Returns the size as an integer or None if no size is found.
    """
    size_patterns = [
        r"[Ss]ize:?\s*(\d+)",
        r"\*\*[Ss]ize\*\*:?\s*(\d+)",
        r"\*\*[Ss]ize:\*\*\s*(\d+)",  # Pattern for **Size:** 24
        r"[Cc]over\s+[Ss]ize:?\s*(\d+)",
        r"[Ss]ize\s+of\s+[Cc]over:?\s*(\d+)",
        r"[Ss]ize\s+is:?\s*(\d+)",
        r"[Hh]as\s+[Ss]ize:?\s*(\d+)",
        r"[Cc]over\s+of\s+size\s+(\d+)",
        r"[Vv]ertex\s+[Cc]over\s+(?:has|with|of)\s+(\d+)\s+(?:nodes|vertices)",
        r"-\s*(?:\*\*)?[Ss]ize:?(?:\*\*)?\s*(\d+)"  # Pattern for: - Size: 6
    ]

    for pattern in size_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                size = int(match.group(1))
                return size if size > 0 else None
            except ValueError:
                continue

    # Fallback: Use the length of cover list if provided
    if cover is not None:
        return len(cover) if cover else None

    return None


def parse_vertex_cover(response: str) -> Dict:
    """
    Extract minimum vertex cover information from LLM response.
    Returns a dictionary with 'cover' and 'size' keys.
    """
    # Parse vertex cover nodes
    cover = parse_vertex_cover_nodes(response)

    # Parse vertex cover size
    size = parse_vertex_cover_size(response, cover)

    return {
        "cover": cover,
        "size": size
    }
# !MAX CLIQUE


def count_clique_edges(clique_nodes: List[str], lst_file: str) -> int:
    """
    Count how many edges exist between the nodes in the predicted clique
    by checking the .lst file

    Args:
        clique_nodes: List of node IDs in the predicted clique
        lst_file: Path to the .lst file containing the graph structure

    Returns:
        Number of edges that exist between the clique nodes
    """
    # Convert nodes to strings and create a set for O(1) lookup
    clique_set = set(str(n) for n in clique_nodes)
    edges = set()

    try:
        with open(lst_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(':')
                if len(parts) != 2:
                    continue

                source = parts[0].strip()
                # Only process if this node is in our clique
                if source in clique_set:
                    targets = parts[1].strip().split()

                    # Count edges to other clique nodes
                    for target in targets:
                        if target in clique_set:
                            # Sort nodes to ensure consistent edge representation
                            edge = tuple(sorted([source, target]))
                            edges.add(edge)

        return len(edges)

    except Exception as e:
        print(f"Error counting clique edges: {e}")
        return 0


def compute_max_clique_accuracy(predicted: Dict, ground_truth: Dict, lst_file: str) -> float:
    """
    Compute accuracy for max clique task using the formula:
    min(gt_size/pred_size, pred_size/gt_size) *
    number_of_edge_in_parsed_clique / \
        (size_max_clique_parsed * (size_max_clique_parsed-1) / 2)

    Args:
        predicted: Dictionary with 'clique' and 'size' keys
        ground_truth: Dictionary with 'clique' and 'size' keys
        lst_file: Path to the .lst file containing the graph structure
    """
    # Handle invalid inputs
    if not predicted or not ground_truth:
        return 0.0

    pred_size = predicted.get('size')
    gt_size = ground_truth.get('size')
    pred_clique = predicted.get('clique')

    # If any required value is missing or size is 0/1, return 0
    if pred_size is None or gt_size is None or not pred_clique or pred_size < 2:
        return 0.0

    try:
        # Compute size ratio
        ratio = min(gt_size/pred_size, pred_size/gt_size)

        # Count actual edges in predicted clique
        actual_edges = count_clique_edges(pred_clique, lst_file)

        # Compute maximum possible edges for a clique of this size
        max_possible_edges = (pred_size * (pred_size - 1)
                              ) // 2

        # Compute clique quality (actual edges / possible edges)
        clique_quality = actual_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        # Compute final accuracy
        accuracy = ratio * clique_quality

        return round(accuracy, 3)

    except Exception as e:
        print(f"Error computing max clique accuracy: {e}")
        return 0.0


def parse_max_clique(response: str) -> Dict:
    """
    Extract maximum clique information from LLM response.
    Returns a dictionary with 'clique' and 'size' keys.
    """
    # Parse clique nodes
    clique = parse_max_clique_nodes(response)

    # Parse clique size
    size = parse_max_clique_size(response, clique)

    return {
        "clique": clique,
        "size": size
    }


def parse_max_clique_nodes(response: str) -> Optional[List[str]]:
    """
    Extract the nodes of the maximum clique from LLM response.
    Returns a list of node IDs or None if no clique is found.
    """
    clique_patterns = [
        r"[Cc]lique:?\s*\[(.*?)\](?:\s*,?\*?)?",  # Handle trailing ],*
        r"\*\*[Cc]lique\*\*:?\s*\[(.*?)\](?:\s*,?\*?)?",
        # Handle **Clique: 1, 2, 3, Size: 3**
        r"\*\*[Cc]lique:?\s*([\d\s,]+)(?=,\s*[Ss]ize:)",
        r"[Mm]aximum\s+[Cc]lique:?\s*\[(.*?)\](?:\s*,?\*?)?",
        r"[Tt]he\s+[Cc]lique\s+is:?\s*\[(.*?)\](?:\s*,?\*?)?",
        r"[Cc]lique:?\s*\{(.*?)\}",
        # Specific pattern for format: "Clique: 5, 6, 7, 8, 9, Size: 5"
        r"[Cc]lique:?\s*([\d\s,]+)(?=,\s*[Ss]ize:)",
        # More specific patterns with boundaries
        r"[Cc]lique:?\s*([\d\s,\*]+)(?=\s+[Ss]ize:|\s*$|\s|,|\.|;)",
        r"[Mm]aximum\s+[Cc]lique:?\s*([\d\s,\*]+)(?=\s*[Ss]ize:|$|\s|,|\.|;)",
        # Generic patterns
        r"[Cc]lique:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Ss]ize)",
        r"[Mm]aximum\s+[Cc]lique:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Ss]ize)",
        r"[Mm]ax\s+[Cc]lique:?\s*((?:\d+(?:\s*[,\*]\s*)?)+)(?=\s|$|,|\.|;|[Ss]ize)"
    ]

    for pattern in clique_patterns:
        match = re.search(pattern, response)
        if match:
            clique_str = match.group(1).strip()

            # Remove "Size:" and anything after if captured
            size_match = re.search(r"(.*?)(?:,\s*[Ss]ize:)", clique_str)
            if size_match:
                clique_str = size_match.group(1).strip()
            elif "size" in clique_str.lower():
                clique_str = clique_str.lower().split("size")[0].strip()

            return clean_number_list(clique_str)

    # If no clique pattern matched, look for cliques mentioned in the text
    step_patterns = [
        r"found.*?cliques?:?\s*\{([\d\s,]+)\}",
        r"cliques?.*?size.*?(\d+).*?:\s*\{([\d\s,]+)\}",
        r"cliques? of size.*?(\d+).*?:\s*\{([\d\s,]+)\}"
    ]

    for pattern in step_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            # Group 2 will have the clique nodes for patterns that mention size first
            if len(match.groups()) > 1 and match.group(2):
                return clean_number_list(match.group(2))
            # Otherwise use group 1
            return clean_number_list(match.group(1))

    return None


def parse_max_clique_size(response: str, clique: Optional[List[str]] = None) -> Optional[int]:
    """
    Extract the size of the maximum clique from LLM response.
    If clique list is provided, uses its length as a fallback if no explicit size is found.
    Returns the size as an integer or None if no size is found.
    """
    # First check for an explicit size in the final statement with higher priority
    final_size_patterns = [
        # **Clique: [1,2,3], Size: 5**
        r"\*\*[Cc]lique:.*?[Ss]ize:\s*(\d+)\*\*\s*$",
        # Clique: [1,2,3], Size: 5 at end
        r"[Cc]lique:.*?[Ss]ize:\s*(\d+)\s*$",
        r"[Ss]ize:\s*(\d+)\s*$",                      # Size: 5 at end
        r"\*\*[Ss]ize:\s*(\d+)\*\*\s*$"               # **Size: 5** at end
    ]

    # Try to match final statement patterns with high priority
    for pattern in final_size_patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            try:
                size = int(match.group(1))
                return size if size > 0 else None
            except ValueError:
                continue

    # Look for bold format clique and size
    bold_pattern = r"\*\*[Cc]lique:.*?\*\*.*?\*\*[Ss]ize:\s*(\d+)\*\*"
    match = re.search(bold_pattern, response)
    if match:
        try:
            size = int(match.group(1))
            return size if size > 0 else None
        except ValueError:
            pass

    # General size patterns with lower priority
    size_patterns = [
        r"[Ss]ize:?\s*(\d+)",
        r"\*\*[Ss]ize\*\*:?\s*(\d+)",
        r"[Cc]lique\s+[Ss]ize:?\s*(\d+)",
        r"[Ss]ize\s+of\s+[Cc]lique:?\s*(\d+)",
        r"[Ss]ize\s+is:?\s*(\d+)",
        r"[Hh]as\s+[Ss]ize:?\s*(\d+)",
        r"[Cc]lique\s+of\s+size\s+(\d+)",
        r"[Mm]aximum\s+[Cc]lique\s+(?:has|with|of)\s+(\d+)\s+(?:nodes|vertices)"
    ]

    for pattern in size_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                size = int(match.group(1))
                return size if size > 0 else None
            except ValueError:
                continue

    # Look for mentions of clique size in the analysis
    step_size_patterns = [
        r"found.*?cliques? of size\s+(\d+)",
        r"cliques? of size\s+(\d+)",
        r"size-(\d+) cliques?"
    ]

    for pattern in step_size_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                size = int(match.group(1))
                return size if size > 0 else None
            except ValueError:
                continue

    # Fallback: Use the length of clique list if provided
    if clique is not None:
        return len(clique) if clique else None

    return None

# !SAVE FUNCTIONS


def save_ground_truth(output_folder: Path, data: List[Dict], experiment_info: Dict):
    """Save ground truth data with experiment metadata"""
    filename = f"ground_truth_{experiment_info['task']}_{experiment_info['graph_type']}_{experiment_info['type']}.csv"
    df = pd.DataFrame(data)
    df.to_csv(output_folder / filename, index=False)


def save_results(output_folder: Path, results: List[Dict], experiment_info: Dict):
    """Save experiment results with metadata"""
    thick_tag = "_thick" if experiment_info.get(
        'use_thick_drawings', False) else ""

    filename = f"results_{experiment_info['task']}_{experiment_info['folder_name']}{thick_tag}_{experiment_info['type']}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_folder / filename, index=False)

# !PARSER


def parse_llm_response(response: str, task_name: str, graph_type: str) -> Dict:
    """Extract structured data from LLM text response based on task type"""
    if task_name == "shortest_path":
        result = parse_shortest_path(response)
        return result

    elif task_name == "common_neighbours":
        # Use the dedicated parser functions
        neighbours = parse_common_neighbors(response)
        number = parse_common_neighbors_count(response, neighbours)

        return {
            "neighbours": neighbours,
            "number": number
        }

    elif task_name == "max_clique":
        # Use the dedicated parser function
        return parse_max_clique(response)

    elif task_name == "min_vertex_cover":
        # Use the dedicated parser function
        return parse_vertex_cover(response)

    elif task_name == "adjacency_list":
        # Use the dedicated parser function
        result = parse_al(response)
        return result if result else {"adjacency_list": ""}

    return {}

# Helper function used by the parser


def clean_number_list(number_str: str) -> List:
    """
    Clean and convert a string of numbers into a list.
    Handles various formats including asterisks, commas, spaces, and brackets.

    Example inputs:
    "1, 2, 3,* 4,*"
    "[1, 2, 3,* 4],*"
    "[1,2,3,4]"
    "1 2 3 4"
    "{1, 2, 3, 4}"
    """
    # First, remove the trailing asterisk if it exists after a closing bracket
    number_str = re.sub(r'\],\*\s*$', ']', number_str)

    # Remove common delimiters and unwanted characters
    for char in "[]{}*":
        number_str = number_str.replace(char, '')

    # Split by comma and clean
    numbers = [n.strip() for n in number_str.replace(' ', ',').split(',')]

    # Filter out empty strings
    return [n for n in numbers if n.strip()]


def compute_relative_accuracy(predicted: float, ground_truth: float) -> float:
    """Calculate accuracy using relative error formula: max(1 - |correct - actual|/correct, 0)"""
    if ground_truth == 0:
        return 1.0 if predicted == 0 else 0.0

    relative_error = abs(ground_truth - predicted) / ground_truth
    return round(max(1 - relative_error, 0), 3)


def compute_accuracy(predicted: Dict, ground_truth: Dict, task_name: str = None, lst_file: str = None) -> float:
    """Calculate accuracy between predicted and ground truth values"""
    if not predicted or not ground_truth:
        return 0.0

    if task_name == "shortest_path" and lst_file:
        return compute_shortest_path_accuracy(predicted, ground_truth, lst_file)

    if task_name == "min_vertex_cover" and lst_file:
        return compute_vertex_cover_accuracy(predicted, ground_truth, lst_file)

    if task_name == "max_clique" and lst_file:
        return compute_max_clique_accuracy(predicted, ground_truth, lst_file)

    if task_name == 'common_neighbours':
        return compute_average_node_jaccard_similarity_cn(predicted, ground_truth)

    elif "adjacency_list" in ground_truth:
        pred_list = predicted.get("adjacency_list", "")
        gt_list = ground_truth.get("adjacency_list", "")

        if not pred_list or not gt_list:
            return 0.0

        return compute_average_node_jaccard_similarity_adjlist(pred_list, gt_list)

    return 0.0
