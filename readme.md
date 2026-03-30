# GD4LLM

This project runs experiments to evaluate Large Language Models (LLMs) on graph analysis tasks. It supports both visual and textual graph analysis with different prompting strategies and reasoning approaches.

## Features

- Visual analysis using graph layouts
- Textual analysis using adjacency list representations
- Multiple prompting strategies (zero-shot and few-shot)
- Different reasoning approaches (standard, chain-of-thought, and spell-out-adjacency-list)
- Automated experiment management and result collection


## Project Structure

```
.
├── src/                    # Core source code
│   ├── main.py            # Main experiment runner
│   ├── graph_tasks.py     # Task definitions and parameters
│   ├── text_analyzer.py   # Text-based graph analysis
│   ├── image_analyzer.py  # Image-based graph analysis
│   ├── mixed_analyzer.py  # Combined visual and textual analysis
│   ├── utils.py          # Utility functions
│   ├── model_selector.py # LLM model selection and configuration
│   ├── tasks/            # Task-specific configurations and prompts
│   ├── examples/         # Example graphs and test cases
│   └── output/           # Experiment results and logs
├── benchmarks/           # Graph datasets and representations
│   ├── drawings/        # Visual graph representations (standard)
│   └── lst/            # Adjacency list representations
├── layout/              # Graph layout algorithms and visualization
    └── ogdf.ipynb      # OGDF layout algorithm implementation

```

# TVCG Replicability Stamp

To support the TVCG Replicability Stamp, we provide an `OPENAI_API_KEY` with limited credits sufficient to run and validate the code.

Our implementation also supports **Anthropic**, **Gemini**, and **LLama**. However, API keys for these services are not provided.

---

## Benchmark Description

For the sake of simplicity, we include a predefined evaluation task:

- **Task**: Shortest path on a single graph  
- **Graph drawings**: `benchmarks/drawings/replicability-test/`  
- **Adjacency lists**: `benchmarks/lst/replicability-test/`  

The evaluation runs the model (**gpt-4o**) across three input modalities:
- Text-only
- Visual-only
- Mixed (text + image)

The output consists of logs capturing model responses for each modality.

Here we include a list of all the possible experimental configurations:
- `tasks: [adjacency_list, common_neighbours, max_clique, min_vertex_cover, shortest_path]`
- `benchmarks: [clique, general, improved_drawings, large_graphs, replicability_test, vertex_cover]`
- `models: [gpt-4o, claude-sonnet-3.7-latest, meta/llama-4-maverick-17b-128e-instruct-maas]`

---

## Run the project

From the project root:
```bash
chmod +x run.sh
./run.sh
```
This will:
- Load environment variables from config.txt
- Export OPENAI_API_KEY
- Execute the main Python pipeline

### Manual Execution (Optional)
Alternatively, run:
```bash
source ./setup.sh
python src/main.py
```

## Output
Results are saved in a log format in: `src/output/shortest_path/`. Note: we do not have any images produced in this code.

Each run produces CSV files containing:

- Ground truth values
- Model predictions
- Accuracy metrics
- Latency measurements
- Token usage statistics

### Optional after the experiments:
To remove the api key from your shell:

```bash
unset OPENAI_API_KEY
sed -i '/^export OPENAI_API_KEY=/d' ~/.bashrc
```
