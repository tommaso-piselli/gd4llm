from base_analyzer import BaseAnalyzer
from utils import encode_image, parse_llm_response, compute_accuracy
from pathlib import Path
from typing import List, Dict
import time
from langchain_core.messages import HumanMessage, SystemMessage


class MixedTaskRunner(BaseAnalyzer):
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

    def _create_messages(self, parameters: Dict, graph_text: str,
                         image_data: str, prompt_type: str, reasoning_type: str) -> List:
        """Create messages with both text and image content"""
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
                examples_content.extend([
                    {"type": "text",
                        "text": f"Graph:\n{example['graph_text']}\n"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{example['image_data']}"}
                    },
                    {"type": "text",
                        "text": f"\nSolution: {example['solution']}\n"}
                ])

            # Add examples as a single human message
            if examples_content:
                messages.append(HumanMessage(content=examples_content))

        # Add final message with the actual task
        messages.append(
            HumanMessage(content=[
                {"type": "text", "text": f"Graph:\n{graph_text}\n{formatted_user_message}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                }
            ])
        )

        return messages

    def _load_examples(self, prompt_type: str, reasoning_type: str) -> List[Dict]:
        """Load examples with both text and image data"""
        try:
            task_config = self.config[self.settings.task_name]
            examples = task_config["prompts"]["few_shots"][reasoning_type]["examples"]

            formatted_examples = []
            for example in examples:
                with open(Path(example["graph_data"]), 'r') as f:
                    graph_text = f.read()
                image_data = encode_image(Path(example["image_data"]))

                formatted_examples.append({
                    "graph_text": graph_text,
                    "image_data": image_data,
                    "solution": example["task_solution"]
                })
            return formatted_examples
        except Exception as e:
            print(f"Error loading examples: {str(e)}")
            return []

    def process_graph_with_stimuli(self, graph_file, image_files, stimuli, ground_truths):
        graph_name = graph_file.stem
        all_results = []

        try:
            with open(graph_file, 'r') as f:
                graph_text = f.read()

            for image_file in image_files:
                layout_type = self._determine_layout_type(str(image_file))
                print(f"\nTesting layout: {layout_type}")

                image_data = encode_image(image_file)

                for params in stimuli:
                    for prompt_type in self.strategies:
                        for reasoning_type in self.strategies[prompt_type]:
                            print(
                                f"Running {prompt_type} {reasoning_type} experiment")

                            try:
                                messages = self._create_messages(
                                    params, graph_text, image_data, prompt_type, reasoning_type
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
                                        lst_file=str(graph_file)
                                    ),
                                    "latency": end_time - start_time,
                                    "input_tokens": response.usage_metadata.get("input_tokens", 0),
                                    "output_tokens": response.usage_metadata.get("output_tokens", 0),
                                    "total_tokens": response.usage_metadata.get("input_tokens", 0) +
                                    response.usage_metadata.get(
                                        "output_tokens", 0)
                                })

                            except Exception as e:
                                print(f"Error in experiment: {str(e)}")
                                continue

            return all_results

        except Exception as e:
            print(f"Error processing graph {graph_name}: {str(e)}")
            return []
