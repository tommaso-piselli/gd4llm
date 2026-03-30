from base_analyzer import BaseAnalyzer
from utils import encode_image, parse_llm_response, compute_accuracy
from pathlib import Path
from typing import List, Dict
import time
from langchain_core.messages import HumanMessage, SystemMessage


class VisualTaskRunner(BaseAnalyzer):
    def __init__(self, settings, config_path: Path, output_folder: Path):
        # First, call the parent constructor
        super().__init__(settings, config_path, output_folder)

        # Then override the strategies dictionary to include 'soal'
        if self.settings.task_name == "adjacency_list":
            self.strategies = {"zero_shot": ["standard", "cot"],
                               "few_shots": ["standard", "cot"]}
        else:
            self.strategies = {
                "zero_shot": ["standard", "cot", "soal"],
                "few_shots": ["standard", "cot", "soal"]
            }

    def _determine_layout_type(self, file_path: str) -> str:
        """Determine the layout type from the file path"""
        if "IMPROVED" in file_path:
            return "IMPROVED"
        elif "FMMM" in file_path:
            return "FMMM"
        elif self.settings.graph_type == "di":
            return "SL"
        else:
            return "ORTHO"

    def _create_messages(self, parameters: Dict, image_data: str,
                         prompt_type: str, reasoning_type: str) -> List:
        """Create messages with only image content, compatible with all models including Claude"""
        task_config = self.config[self.settings.task_name]
        messages = []

        # Add system message
        system_message = task_config["messages"][reasoning_type]["system_message"]
        messages.append(SystemMessage(content=system_message))

        # Format user message
        user_message = task_config["messages"][reasoning_type]["user_message"]
        formatted_user_message = user_message.format(**parameters.params)

        if prompt_type == "few_shots":
            examples = self._load_examples(prompt_type, reasoning_type)
            # Combine all examples into a single message
            examples_content = []
            for example in examples:
                examples_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{example['image']}"}
                })
                examples_content.append({
                    "type": "text",
                    "text": f"\nSolution: {example['solution']}\n"
                })

            # Add examples as a single human message
            if examples_content:
                messages.append(HumanMessage(content=examples_content))

        # Add final message with the actual task
        final_message_content = [
            {"type": "text", "text": formatted_user_message},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            }
        ]
        messages.append(HumanMessage(content=final_message_content))

        return messages

    def _load_examples(self, prompt_type: str, reasoning_type: str) -> List[Dict]:
        """Load examples with only image data"""
        try:
            task_config = self.config[self.settings.task_name]
            examples = task_config["prompts"]["few_shots"][reasoning_type]["examples"]

            formatted_examples = []
            for example in examples:
                example_image = encode_image(Path(example["image_data"]))
                formatted_examples.append({
                    "image": example_image,
                    "solution": example["task_solution"]
                })
            return formatted_examples
        except Exception as e:
            print(f"Error loading examples: {str(e)}")
            return []

    def process_graph_with_stimuli(self, image_files: List[Path], stimuli: List, ground_truths: List):
        all_results = []

        for graph_file in image_files:
            layout_type = self._determine_layout_type(str(graph_file))
            print(f"\nTesting layout: {layout_type}")

            image_data = encode_image(graph_file)

            for params in stimuli:
                for prompt_type in self.strategies:
                    for reasoning_type in self.strategies[prompt_type]:
                        print(
                            f"Running {prompt_type} {reasoning_type} experiment")

                        try:
                            messages = self._create_messages(
                                params, image_data, prompt_type, reasoning_type
                            )

                            start_time = time.time()
                            response = self.model.invoke(messages)
                            end_time = time.time()

                            parsed_response = parse_llm_response(
                                response.content,
                                self.settings.task_name,
                                self.settings.graph_type
                            )

                            ground_truth = next(
                                gt["ground_truth"] for gt in ground_truths
                                if gt["parameters"] == params.params
                            )

                            graph_name = graph_file.stem.split('-')[0]
                            lst_file = Path(
                                "benchmarks") / "lst" / self.settings.folder_name / f"{graph_name}.lst"

                            all_results.append({
                                "graph": graph_name,
                                "layout": layout_type,
                                "parameters": params.params,
                                "prompt_type": prompt_type,
                                "reasoning_type": reasoning_type,
                                "response": response.content,
                                "parsed_response": parsed_response,
                                "ground_truth": ground_truth,
                                "accuracy": compute_accuracy(
                                    parsed_response,
                                    ground_truth,
                                    task_name=self.settings.task_name,
                                    lst_file=str(lst_file)
                                ),
                                "latency": end_time - start_time,
                                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                                "output_tokens": response.usage_metadata.get("output_tokens", 0),
                                "total_tokens": response.usage_metadata.get("input_tokens", 0) +
                                response.usage_metadata.get("output_tokens", 0)
                            })

                        except Exception as e:
                            print(f"Error in experiment: {str(e)}")
                            continue

        return all_results
