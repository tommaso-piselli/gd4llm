from base_analyzer import BaseAnalyzer
from utils import parse_llm_response, compute_accuracy
from pathlib import Path
from typing import List, Dict
import time
from langchain_core.messages import HumanMessage, SystemMessage


class TextualTaskRunner(BaseAnalyzer):
    def _create_messages(self, parameters: Dict, graph_text: str,
                         prompt_type: str, reasoning_type: str) -> List:
        """Create messages with only text content"""
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
            examples_text = []
            for example in examples:
                examples_text.append(
                    f"Graph:\n{example['graph_data']}\nSolution: {example['solution']}\n"
                )

            if examples_text:
                messages.append(
                    HumanMessage(content="\n".join(examples_text))
                )

        # Add final message
        messages.append(
            HumanMessage(
                content=f"Graph:\n{graph_text}\n{formatted_user_message}")
        )

        return messages

    def _load_examples(self, prompt_type: str, reasoning_type: str) -> List[Dict]:
        """Load examples with only text data"""
        try:
            task_config = self.config[self.settings.task_name]
            examples = task_config["prompts"]["few_shots"][reasoning_type]["examples"]

            formatted_examples = []
            for example in examples:
                with open(Path(example["graph_data"]), 'r') as f:
                    graph_text = f.read()

                formatted_examples.append({
                    "graph_data": graph_text,
                    "solution": example["task_solution"]
                })
            return formatted_examples
        except Exception as e:
            print(f"Error loading examples: {str(e)}")
            return []

    def process_graph_with_stimuli(self, graph_file: Path, stimuli: List, ground_truths: List):
        graph_name = graph_file.stem
        print(f"\nProcessing graph: {graph_name}")

        try:
            with open(graph_file, 'r') as f:
                graph_text = f.read()

            all_results = []

            for params in stimuli:
                for prompt_type in self.strategies:
                    for reasoning_type in self.strategies[prompt_type]:
                        print(
                            f"Running {prompt_type} {reasoning_type} experiment")

                        try:
                            messages = self._create_messages(
                                params, graph_text, prompt_type, reasoning_type
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
                                response.usage_metadata.get("output_tokens", 0)
                            })

                        except Exception as e:
                            print(f"Error in experiment: {str(e)}")
                            continue

            return all_results

        except Exception as e:
            print(f"Error processing graph {graph_name}: {str(e)}")
            return []
