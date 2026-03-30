from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Literal, Optional, Set
from image_analyzer import VisualTaskRunner
from text_analyzer import TextualTaskRunner
from mixed_analyzer import MixedTaskRunner
from graph_tasks import TaskFactory, load_graph
from utils import save_results
from model_selector import ModelFamily, ModelConfig


@dataclass
class ModelSettings:
    family: ModelFamily
    name: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    def to_config(self) -> ModelConfig:
        return ModelConfig(
            family=self.family,
            name=self.name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


@dataclass
class ExperimentSettings:
    task_name: str
    graph_type: Literal["di", "undi"]
    folder_name: str
    num_stimuli: int = 2
    model: ModelSettings = ModelSettings(
        family=ModelFamily.GPT, name="gpt-4o", temperature=0.0)
    use_thick_drawings: bool = False
    improved_drawings: bool = False
    seed: Optional[int] = None  # Added seed parameter
    # New parameter to select analysis types
    analysis_types: Set[Literal["visual", "textual", "mixed"]] = None

    def __post_init__(self):
        # Default to all available analysis types if none specified
        if self.analysis_types is None:
            if self.task_name == "adjacency_list":
                # For adjacency_list, only visual analysis is available
                self.analysis_types = {"visual"}
            else:
                # For other tasks, all three types are available
                self.analysis_types = {"visual", "textual", "mixed"}


class ExperimentPipeline:
    def __init__(self, settings: ExperimentSettings, config_base: Path, output_folder: Path):
        self.settings = settings
        self.output_folder = output_folder
        self.task = TaskFactory.create_task(
            settings.task_name, settings.graph_type)

        # Initialize runners based on selected analysis types
        self.visual_runner = None
        self.textual_runner = None
        self.mixed_runner = None

        # Initialize visual runner if selected
        if "visual" in settings.analysis_types:
            visual_config = config_base / \
                f"{settings.task_name}_{settings.graph_type}_visual.json"
            self.visual_runner = VisualTaskRunner(
                settings, visual_config, output_folder)

        # For all other tasks, initialize textual and mixed runners if selected
        if settings.task_name != "adjacency_list":
            if "textual" in settings.analysis_types:
                textual_config = config_base / \
                    f"{settings.task_name}_{settings.graph_type}_textual.json"
                self.textual_runner = TextualTaskRunner(
                    settings, textual_config, output_folder)

            if "mixed" in settings.analysis_types:
                mixed_config = config_base / \
                    f"{settings.task_name}_{settings.graph_type}_mixed.json"
                self.mixed_runner = MixedTaskRunner(
                    settings, mixed_config, output_folder)

    def _get_image_files(self, lst_file: Path) -> List[Path]:
        """Get the corresponding image files for a given .lst file"""
        base_name = lst_file.stem

        # Determine which drawings folder to use based on settings
        base_folder = "drawings_thick" if self.settings.use_thick_drawings else "drawings"
        print(f"Using drawings folder: {base_folder}")

        # Try both absolute and relative paths for the drawings folder
        possible_paths = [
            Path("./benchmarks") / base_folder /
            self.settings.folder_name,  # Relative path
            Path("/benchmarks") / base_folder /
            self.settings.folder_name,   # Absolute path
            lst_file.parent.parent.parent / base_folder /
            self.settings.folder_name  # Path relative to lst file
        ]

        drawings_folder = None
        for path in possible_paths:
            if path.exists():
                drawings_folder = path
                print(f"Found drawings folder at: {drawings_folder}")
                break

        if not drawings_folder:
            print(f"Warning: Drawings folder not found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            return []

        # Define base patterns depending on graph type
        base_patterns = {
            'fmmm': f"{base_name}-FMMM*.png",
            'ortho': f"{base_name}-ORTHO*.png" if self.settings.graph_type == "undi" else f"{base_name}-SL*.png",
            'improved': f"{base_name}-IMPROVED*.png"
        }

        # Collect all matching files based on settings
        files = []

        # Always look for FMMM and ORTHO/SL patterns
        files.extend(drawings_folder.glob(base_patterns['fmmm']))
        files.extend(drawings_folder.glob(base_patterns['ortho']))

        # Add IMPROVED pattern if enabled
        if self.settings.improved_drawings:
            files.extend(drawings_folder.glob(base_patterns['improved']))

        # Debug print of found files
        found_files = sorted(files)
        if found_files:
            print(f"Found {len(found_files)} drawing files:")
            for f in found_files:
                print(f"  - {f.name}")
        else:
            print("No drawing files found matching the patterns:")
            for pattern in base_patterns.values():
                print(f"  - {pattern}")

        return found_files

    def generate_experiment_data(self, graph_file: Path):
        """Generate experiment data with additional debug logging"""
        print(f"Loading graph from file: {graph_file.absolute()}")
        G = load_graph(graph_file, self.settings.graph_type)
        print(
            f"Graph loaded successfully. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        # Pass the seed to generate_stimuli
        stimuli = self.task.generate_stimuli(
            G, self.settings.num_stimuli, seed=self.settings.seed)

        if self.settings.seed is not None:
            print(f"Using seed {self.settings.seed} for stimulus generation")
        print(f"Generated {len(stimuli)} stimuli")

        ground_truths = []
        for params in stimuli:
            try:
                if self.settings.task_name in ["min_vertex_cover", "max_clique"]:
                    graph_name = graph_file.stem
                    truth = self.task.compute_ground_truth(graph_name, params)
                else:
                    print(
                        f"Computing ground truth for graph: {graph_file.stem}")
                    truth = self.task.compute_ground_truth(G, params)

                ground_truths.append({
                    "graph": graph_file.stem,
                    "parameters": params.params,
                    "ground_truth": truth
                })
                print(
                    f"Ground truth computed successfully for {graph_file.stem}")

            except Exception as e:
                print(f"Error computing ground truth: {str(e)}")
                print(f"Graph file: {graph_file}")
                print(f"Parameters: {params}")
                raise

        return stimuli, ground_truths

    def run_experiments(self, data_folder: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        print(f"\nStarting {self.settings.task_name} experiments")
        print(
            f"Using {'thick' if self.settings.use_thick_drawings else 'regular'} drawings")
        print(
            f"Enabled analysis types: {', '.join(self.settings.analysis_types)}")

        if self.settings.seed is not None:
            print(
                f"Using seed {self.settings.seed} for reproducible stimulus generation")

        graph_files = sorted(data_folder.glob("*.lst"))
        total_graphs = len(graph_files)
        print(f"Found {total_graphs} graphs to process")
        print(f"Data folder: {data_folder}")

        visual_results = []
        textual_results = []
        mixed_results = []

        for i, graph_file in enumerate(graph_files, 1):
            try:
                print(f"\nProcessing graph {i} of {total_graphs}")
                print(f"Graph file: {graph_file}")

                # Generate data once
                stimuli, ground_truths = self.generate_experiment_data(
                    graph_file)
                print(f"Generated {len(stimuli)} stimuli")

                # Run visual experiments if enabled
                if "visual" in self.settings.analysis_types and self.visual_runner:
                    print("\n> Running visual analysis...")
                    image_files = self._get_image_files(graph_file)

                    if image_files:
                        print(f"Image files: {[f.name for f in image_files]}")
                        vis_results = self.visual_runner.process_graph_with_stimuli(
                            image_files, stimuli, ground_truths
                        )
                        visual_results.extend(vis_results)
                    else:
                        print("Skipping visual analysis - no image files found")

                # Only run textual and mixed if task is not adjacency_list
                if self.settings.task_name != "adjacency_list":
                    # Run textual experiments if enabled
                    if "textual" in self.settings.analysis_types and self.textual_runner:
                        print("\n> Running textual analysis...")
                        text_results = self.textual_runner.process_graph_with_stimuli(
                            graph_file, stimuli, ground_truths
                        )
                        textual_results.extend(text_results)

                    # Run mixed experiments if enabled
                    if "mixed" in self.settings.analysis_types and self.mixed_runner:
                        print("\n> Running mixed analysis...")
                        image_files = image_files if 'image_files' in locals(
                        ) else self._get_image_files(graph_file)
                        if image_files:
                            mixed_results.extend(
                                self.mixed_runner.process_graph_with_stimuli(
                                    graph_file, image_files, stimuli, ground_truths
                                )
                            )
                        else:
                            print("Skipping mixed analysis - no image files found")

            except Exception as e:
                print(f"Error processing graph {graph_file.stem}: {str(e)}")
                print(f"Error details: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                continue

        print("\nExperiment summary:")
        print(f"- Total graphs processed: {total_graphs}")

        # Save results only for enabled analysis types
        if "visual" in self.settings.analysis_types:
            print(f"- Visual results: {len(visual_results)}")
            metadata = {
                "task": self.settings.task_name,
                "folder_name": self.settings.folder_name,
                "type": "visual",
                "use_thick_drawings": self.settings.use_thick_drawings
            }
            if self.settings.seed is not None:
                metadata["seed"] = self.settings.seed
            save_results(self.output_folder, visual_results, metadata)

        if self.settings.task_name != "adjacency_list":
            if "textual" in self.settings.analysis_types:
                print(f"- Textual results: {len(textual_results)}")
                textual_metadata = {
                    "task": self.settings.task_name,
                    "folder_name": self.settings.folder_name,
                    "type": "textual",
                    "use_thick_drawings": self.settings.use_thick_drawings
                }
                if self.settings.seed is not None:
                    textual_metadata["seed"] = self.settings.seed
                save_results(self.output_folder,
                             textual_results, textual_metadata)

            if "mixed" in self.settings.analysis_types:
                print(f"- Mixed results: {len(mixed_results)}")
                mixed_metadata = {
                    "task": self.settings.task_name,
                    "folder_name": self.settings.folder_name,
                    "type": "mixed",
                    "use_thick_drawings": self.settings.use_thick_drawings
                }
                if self.settings.seed is not None:
                    mixed_metadata["seed"] = self.settings.seed
                save_results(self.output_folder, mixed_results, mixed_metadata)

        return visual_results, textual_results, mixed_results


class ExperimentManager:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings
        self.config_base = self._get_config_base()
        self.data_folder = Path("./benchmarks/lst") / settings.folder_name
        self.drawings_folder = self._get_drawings_folder()
        self.output_folder = self._setup_output_folders()

    def _get_config_base(self) -> Path:
        return Path("./src/tasks") / self.settings.task_name

    def _get_drawings_folder(self) -> Path:
        base_folder = "drawings_thick" if self.settings.use_thick_drawings else "drawings"
        return Path("./benchmarks") / base_folder / self.settings.folder_name

    def _setup_output_folders(self) -> Path:
        output_base = Path("./src/output") / self.settings.task_name
        output_base.mkdir(parents=True, exist_ok=True)
        return output_base

    def _validate_paths(self):
        if not self.config_base.exists():
            raise FileNotFoundError(
                f"Config directory not found: {self.config_base}")

        # Only validate configs for the selected analysis types
        if "visual" in self.settings.analysis_types:
            visual_config = self.config_base / \
                f"{self.settings.task_name}_{self.settings.graph_type}_visual.json"
            if not visual_config.exists():
                raise FileNotFoundError(
                    f"Visual config file not found: {visual_config}")

        # Only check textual and mixed configs if not adjacency_list task
        if self.settings.task_name != "adjacency_list":
            if "textual" in self.settings.analysis_types:
                textual_config = self.config_base / \
                    f"{self.settings.task_name}_{self.settings.graph_type}_textual.json"
                if not textual_config.exists():
                    raise FileNotFoundError(
                        f"Textual config file not found: {textual_config}")

            if "mixed" in self.settings.analysis_types:
                mixed_config = self.config_base / \
                    f"{self.settings.task_name}_{self.settings.graph_type}_mixed.json"
                if not mixed_config.exists():
                    raise FileNotFoundError(
                        f"Mixed config file not found: {mixed_config}")

        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.data_folder}")

        if "visual" in self.settings.analysis_types or "mixed" in self.settings.analysis_types:
            if not self.drawings_folder.exists():
                raise FileNotFoundError(
                    f'Drawings folder not found: {self.drawings_folder}')

    def run(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        self._validate_paths()

        print(f"\nStarting {self.settings.task_name} task experiments")
        print(f"Configuration: {self.settings.graph_type} graphs")
        print(
            f"Running analysis types: {', '.join(self.settings.analysis_types)}")
        if self.settings.seed is not None:
            print(f"Using seed: {self.settings.seed}")

        try:
            pipeline = ExperimentPipeline(
                self.settings,
                self.config_base,
                self.output_folder
            )

            visual_results, textual_results, mixed_results = pipeline.run_experiments(
                self.data_folder)

            print("\nExperiment execution completed successfully!")
            return visual_results, textual_results, mixed_results

        except Exception as e:
            print(f"Error during experiment execution: {str(e)}")
            raise
