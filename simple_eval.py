import lm_eval
from evaluation.llamppl_inference_adapter import LLaMPPLInferenceModel
import json
import os

NUM_WORKERS = 2

lm_obj = LLaMPPLInferenceModel(num_particles=3, beam_factor=1, num_tokens=50)

task_manager = lm_eval.tasks.TaskManager(include_path="/path/to/lm-evaluation-harness/lm_eval/tasks/")

results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["mmlu_smc_regex"],
    num_fewshot=0,
    batch_size=1,
    task_manager=task_manager,
    device="cpu",
    limit=1
)

output_dir = "results"
output_filename = os.path.join(output_dir, "run_data.json")

os.makedirs(output_dir, exist_ok=True)

with open(output_filename, "w") as file:
    json.dump(results, file, indent=4)
print(f"Evaluation results successfully saved to {output_filename}")