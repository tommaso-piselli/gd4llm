from experiment_orchestrator import ExperimentSettings, ModelSettings, ExperimentManager
from model_selector import ModelFamily

if __name__ == "__main__":
    # Configure model settings
    model_settings = ModelSettings(
        family=ModelFamily.GPT,  # or ModelFamily.CLAUDE or ModelFamily.GPT
        # gpt-4o claude-3-7-sonnet-latest meta/llama-4-maverick-17b-128e-instruct-maas
        name="gpt-4o",
        temperature=0.0,
        max_tokens=4096,  # high enough, needed for Claude
    )

    # Configure experiment settings
    settings = ExperimentSettings(
        task_name="shortest_path",
        graph_type="undi",
        folder_name="replicability_test",
        num_stimuli=1,
        model=model_settings,
        use_thick_drawings=False,
        improved_drawings=False,
        seed=42,

        # Select only visual analysis
        analysis_types={"visual", "textual", "mixed"}  # Run all (default)

    )

    try:
        manager = ExperimentManager(settings)
        visual_results, textual_results, mixed_results = manager.run()

        # Note: for analysis types that weren't run, the corresponding results will be empty lists
        print(f"Visual results count: {len(visual_results)}")
        print(f"Textual results count: {len(textual_results)}")
        print(f"Mixed results count: {len(mixed_results)}")

    except KeyboardInterrupt:
        print("\nExperiment execution interrupted by user")
    except Exception as e:
        print(f"Experiment execution failed: {str(e)}")
        raise
