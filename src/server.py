import asyncio
import logging
import math
import numpy as np
import os
import time
import torch
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from llamppl import Model, LMContext, CachedCausalLM, TokenCategorical, Token, smc_steer
from typing import List, Dict, Any
from util.request_model import GenerationRequest

MODEL_NAME = "NousResearch/Hermes-3-Llama-3.2-3B"
WORKER_ID = int(os.environ.get("WORKER_ID", 0))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

lm_models: List[CachedCausalLM] = []
current_model_idx: int = 0
model_locks: List[asyncio.Lock] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lm_models, model_locks

    local_gpu_id = 0
    logger.info(f"Worker {WORKER_ID} initializing model on local GPU {local_gpu_id}")

    kwargs = {
        "engine_opts": {
            "gpu_memory_utilization": 0.85,
            "max_model_len": 1024,
            "enforce_eager": True
        }
    }

    logger.info(f"Loading model '{MODEL_NAME}' on local GPU {local_gpu_id}...")
    try:
        lm = CachedCausalLM.from_pretrained(MODEL_NAME, backend='vllm', **kwargs)
        lm.batch_size = 1

        lm_models = [lm]
        model_locks = [asyncio.Lock()]

        logger.info(f"Warming up model on local GPU {local_gpu_id}...")
        dummy_model = FixedLengthSentenceModel(lm=lm, prompt="Hello, world", num_tokens=20)
        await smc_steer(dummy_model, 1, 1)
        logger.info(f"Model warmup complete on local GPU {local_gpu_id}")
    except Exception as e:
        logger.error(f"Failed to load model on worker {WORKER_ID}: {e}")
        raise

    yield

    logger.info(f"Shutting down model on worker {WORKER_ID} on local GPU {local_gpu_id}...")
    for lm_instance in lm_models:
        del lm_instance

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class FixedLengthSentenceModel(Model):
    """Generates sentences of fixed length ending with a period using SMC steering."""

    def __init__(self, lm: CachedCausalLM, prompt: str, num_tokens: int = 10, temperature: float = 1.0):
        super().__init__()

        self.lm = lm
        self.context = LMContext(lm, prompt, temperature)
        self.num_tokens = num_tokens
        self.generated_tokens: List[Token] = []
        self.max_tokens = num_tokens
        self.eos_token = lm.tokenizer.eos_token_id

        self.period_tokens = set()
        if self.lm.vocab:
            for i, token_str in enumerate(self.lm.vocab):
                if token_str and token_str.endswith('.'):
                    self.period_tokens.add(i)
        else:
            logger.warning("LM vocab is empty, period_tokens will not be set")

    async def step(self):
        current_length = len(self.generated_tokens)

        if current_length >= self.num_tokens:
            self.condition(current_length == self.num_tokens)

            if self.generated_tokens and self.generated_tokens[-1].token_id in self.period_tokens:
                self.condition(True)
            else:
                self.condition(False)
            self.finish()
            return

        next_dist = self.context.next_token()

        if current_length == self.num_tokens - 1:
            period_mask = self.period_tokens
            if not period_mask:
                logger.error("period_tokens mask is empty for final token")
                self.condition(False)
                self.finish()
                return
            await self.observe(self.context.mask_dist(period_mask), True)
        else:
            all_token_ids = set(range(len(self.lm.vocab)))
            non_period_mask = all_token_ids - self.period_tokens - {self.lm.tokenizer.eos_token_id}
            if not non_period_mask:
                logger.error("non_period_mask is empty for non-final token")
                self.condition(False)
                self.finish()
                return
            await self.observe(self.context.mask_dist(non_period_mask), True)

        token = await self.sample(next_dist)
        self.generated_tokens.append(token)


start_time = time.time()
total_requests = 0
total_tokens = 0
request_times: List[float] = []

async def get_next_available_model() -> (int, CachedCausalLM, asyncio.Lock):
    if not lm_models:
        raise RuntimeError("Language model not initialized.")
    return 0, lm_models[0], model_locks[0]

async def generate_text(request: GenerationRequest) -> Dict[str, Any]:
    if request.num_particles <= 0 or request.beam_factor <= 0:
        raise ValueError("num_particles and beam_factor must be positive integers.")

    if not lm_models:
        logger.error("Model not initialized. Cannot generate text.")
        raise RuntimeError("Model not initialized.")

    model_idx, lm, lock = await get_next_available_model()

    async with lock:
        with torch.inference_mode():
            model = FixedLengthSentenceModel(
                lm=lm,
                prompt=request.prompt,
                num_tokens=request.num_tokens,
                temperature=request.temperature
            )

            try:
                particles = await smc_steer(model, request.num_particles, request.beam_factor)

                if not particles:
                    logger.warning(f"SMC steering returned no valid particles for request: {request.prompt}")
                    return {"generated_text": "Generation failed: No valid particles found."}

                best_particle = max(particles, key=lambda p: p.weight)
                generated_text = str(best_particle.context)
                return {"generated_text": generated_text}
            except Exception as e:
                logger.error(f"Error during SMC steering for worker {WORKER_ID}: {e}")
                raise

@app.get("/health")
async def health() -> Dict[str, Any]:
    if not lm_models:
        return {"status": "loading"}
    return {
        "status": "ready",
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0
    }

@app.get("/model_info")
async def model_info() -> Dict[str, Any]:
    if not lm_models:
        return {"status": "loading"}

    eos_token_id = lm_models[0].tokenizer.eos_token_id if hasattr(lm_models[0].tokenizer, 'eos_token_id') else None
    max_model_len = lm_models[0].max_model_len if hasattr(lm_models[0], 'max_model_len') else None

    return {
        "eot_token_id": eos_token_id,
        "max_length": max_model_len,
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0
    }

@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    global start_time, total_requests, total_tokens, request_times

    elapsed = time.time() - start_time
    avg_latency = sum(request_times) / len(request_times) if request_times else 0

    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
        "avg_latency": avg_latency,
        "uptime_seconds": elapsed,
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0
    }

@app.post("/generate")
async def generate(request: GenerationRequest) -> Dict[str, Any]:
    global total_requests, total_tokens, request_times

    start_time_req = time.time()
    try:
        result = await generate_text(request)
        duration = time.time() - start_time_req

        request_times.append(duration)
        total_requests += 1
        total_tokens += len(lm_models[0].tokenizer.encode(result.get("generated_text", "")))

        if total_requests % 10 == 0:
            elapsed = time.time() - start_time
            avg_latency = sum(request_times[-100:]) / min(100, len(request_times))
            logger.info(f"Worker {WORKER_ID} | GPU (local) {0} | Requests: {total_requests} | Avg latency (last {min(100, len(request_times))} req): {avg_latency:.4f}s")

        return result
    except ValueError as e:
        logger.error(f"Bad Request in /generate for worker {WORKER_ID}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Server error in /generate for worker {WORKER_ID}: {e}")
        raise HTTPException(status_code=500, detail="Server not ready or misconfigured.")
    except Exception as e:
        logger.error(f"Error in /generate for worker {WORKER_ID}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)