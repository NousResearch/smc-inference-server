# 🌟 SMC Inference Server

Run [Sequential Monte Carlo Steering](https://arxiv.org/abs/2306.03081) as a robust local inference server. This project is designed for exploration, demos, and efficient evaluation with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

For more info, check out our [blog publication](https://smc-blogpost.vercel.app/) covering experiment setups and our takeaways.

---

## 🛠️ Overview

This repository provides a streamlined and extensible foundation for hosting language models that leverage the [llamppl](https://github.com/genlm/llamppl) framework for steerable generation. While `llamppl` integrates with vLLM, this project wraps it with a simple, production-ready API endpoint, making it easier to integrate into your workflows and run benchmarks.

It also includes a distributed framework designed to launch multiple SMC backends across your GPUs, maximizing request processing throughput. This capability aims to enable running SMC benchmarks on extensive datasets like MMLU in a matter of hours.

---

## 🛠️ Setup and Installation

Follow these steps to get the SMC Inference Server up and running.

### Prerequisites

* **Docker**: Ensure Docker is installed and running on your system.
* **CUDA**: Your system should have NVIDIA GPUs and CUDA drivers installed for `vLLM` to function correctly.
* **Python Dependencies (for development/local setup without Docker)**:

    ```bash
    torch 
    vllm 
    llamppl 
    fastapi 
    uvicorn 
    httpx 
    numpy
    ```

    You can use the following to install everything needed:

    ```bash
    pip install -r requirements.txt
    ```

### Step-by-Step Deployment

We recommend using Docker for a consistent and isolated environment and to avoid dependency issues if running the inference server. For demo examples and exploration use pip install with a virtualenv.

1.  **Build the Docker Image**:
    First, build the Docker image for your SMC Inference Server. This command compiles the necessary dependencies and sets up your environment.

    ```bash
    docker build -t llamppl-inference-server .
    ```

2.  **Start Backend Inference Servers**:
    Each backend server will run in its own Docker container, utilizing a dedicated GPU. The `./start_servers.sh` script handles this orchestration. Make sure your `CUDA_VISIBLE_DEVICES` environment variable is correctly configured if you want to assign specific GPUs.

    ```bash
    ./start_servers.sh
    ```
    * **Note**: This script expects your system to have available GPUs and will launch a server for each `WORKER_ID` as defined in the script. Modify the script if you need to control the number of workers or GPU assignments.

    To kill containers, run `docker stop llamppl-worker-X && docker rm llamppl-worker-X`

3.  **Start the Load Balancer**:
    If you want to use multiple GPUs, in a **separate terminal**, start the FastAPI load balancer. This component will distribute incoming requests among your backend inference servers.

    The workers take some time to start as there is a sleep timer in the startup script, so this should be run once they are finished.

    ```bash
    uvicorn load_balancer:app --host 0.0.0.0 --port 8000
    ```
    * The load balancer will be accessible at `http://0.0.0.0:8000`.

---

## 🛠️ Usage

Once the servers are running, you can start sending generation requests.

Sample request:

```bash
curl -X POST "http://0.0.0.0:8000/generate"      -H "Content-Type: application/json"      -d '{
           "prompt": "Here is a short joke:",
           "num_particles": 3,
           "beam_factor": 1,
           "num_tokens": 50
         }'
```

### Testing the SMC Framework for Exploration and Development

To verify that the `llamppl` framework is working as expected (will require pip install) with a simple example:

```bash
python examples/entropy.py
```

## 🛠️ Evaluation

This evaluation framework relies on `generate_until` calls from our custom `LLaMPPLInferenceAdapter` (`evaluation/llamppl_inference_adapter.py`), and uses regex to extract the answer from the response. The sample task definition in `evaluation/tasks/mmlu_smc_regex.yaml` covers how this is set up, though your tasks should be copied to `lm_eval` tasks directory, noted below.

### LM Eval Harness

Clone the lm-evaluation-harness repository following the steps [there](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

To run MMLU with SMC, copy the sample task definition `evaluation/tasks/mmlu_smc_regex.yaml` to the `tasks/llama3/instruct/mmlu/` directory of `lm-evaluation-harness` (/path/to/lm-evaluation-harness/lm_eval/tasks/llama3/instruct/mmlu/ is the format). 

All future task definitions for generative MMLU tasks should go in here as well. Other task definitions can be placed in their respective directories.

Lastly, add the path to the above `tasks` directory to `simple_eval.py`, so `lm-evaluation-harness` can find your new task definition.

Run the server + load balancer, and in another terminal, run `python simple_eval.py` to run the evaluation. Results will be saved to `run_data.json`.

*Note:* this may take a while, recent upgrades to llamppl have required some workarounds to get everything running, this will be updated as issues are resolved.

## 🛠️ Future Work

Development on this project will continue as we explore new constraint designs and optimizations to the framework here. It is both useful for experimenting and for benchmarking at a larger scale, especially on compute clusters with >4 GPUs. 8xH100 was able to finish the entire 15,000 question MMLU set in under 4 hours using this. More updates to come soon!
