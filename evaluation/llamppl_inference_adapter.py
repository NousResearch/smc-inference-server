from lm_eval.api.model import LM
import requests
import asyncio
import aiohttp

class LLaMPPLInferenceModel(LM):
    """Inference adapter for lm-evaluation-harness."""

    def __init__(
        self, 
        server_url="http://localhost:8000",
        num_particles=3,
        beam_factor=1,
        temperature=1.0,
        num_tokens=20,
        batch_size=8,
        **kwargs
    ):
        super().__init__()
        self.server_url = server_url
        self.num_particles = num_particles
        self.beam_factor = beam_factor
        self.temperature = temperature
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code != 200:
                print(f"Warning: Server at {server_url} responded with status {response.status_code}")
            else:
                print(f"Successfully connected to server at {server_url}")
        except Exception as e:
            print(f"Warning: Could not connect to server at {server_url}: {e}")
            print("Make sure the inference server is running")
        
    async def _generate_async(self, request_data, session):
        async with session.post(f"{self.server_url}/generate", json=request_data, timeout=aiohttp.ClientTimeout(total=600)) as response:
            if response.status == 200:
                result = await response.json()
                return result["generated_text"]
            else:
                error_text = await response.text()
                raise Exception(f"Error from server: {error_text}")

    def generate_until(self, reqs):
        async def process_all_requests():
            timeout = aiohttp.ClientTimeout(total=2400)
            connector = aiohttp.TCPConnector(limit=8, force_close=False)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={"Connection": "keep-alive"}
            ) as session:
                all_tasks = []
                for request in reqs:
                    input_text, gen_params = request.args

                    request_data = {
                        "prompt": input_text,
                        "num_tokens": self.num_tokens,
                        "num_particles": self.num_particles,
                        "beam_factor": self.beam_factor,
                        "temperature": self.temperature
                    }
                    
                    all_tasks.append(self._generate_async(request_data, session))
                
                all_results = []
                batch_size = 2

                for i in range(0, len(all_tasks), batch_size):
                    batch = all_tasks[i:i+batch_size]
                    batch_results = await asyncio.gather(*batch)
                    all_results.extend(batch_results)
                                                                            
                return all_results
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(process_all_requests())
        
        return results

    def generate(self, context, max_length, stop=None, temperature=1.0):
        request = {
            "prompt": context,
            "num_tokens": min(max_length, self.num_tokens),
            "num_particles": self.num_particles,
            "beam_factor": self.beam_factor,
            "temperature": temperature if temperature else self.temperature
        }
        
        response = requests.post(
            f"{self.server_url}/generate",
            json=request
        )
        
        if response.status_code == 200:
            generated_text = response.json()["generated_text"]
                        
            return generated_text
        else:
            raise Exception(f"Error from server: {response.text}")

    def loglikelihood(self, reqs):
        batches = [reqs[i:i + self.batch_size] for i in range(0, len(reqs), self.batch_size)]
        results = []
        
        for batch in batches:
            batch_request = {
                "requests": [
                    {"context": context, "continuation": continuation}
                    for context, continuation in batch
                ]
            }
            
            response = requests.post(
                f"{self.server_url}/batch_logprob",
                json=batch_request
            )
            
            if response.status_code == 200:
                batch_results = response.json()["results"]
                results.extend([
                    (result["logprob"], result["is_greedy"]) 
                    for result in batch_results
                ])
            else:
                raise Exception(f"Error from server: {response.text}")
                
        return results

    def loglikelihood_rolling(self, reqs):
        batches = [reqs[i:i + self.batch_size] for i in range(0, len(reqs), self.batch_size)]
        results = []
        
        for batch in batches:
            batch_request = {
                "requests": [
                    {"context": "", "continuation": text[0]}
                    for text in batch
                ]
            }
            
            response = requests.post(
                f"{self.server_url}/batch_logprob_rolling",
                json=batch_request
            )
            
            if response.status_code == 200:
                batch_results = response.json()["results"]
                results.extend([
                    (sum(result["logprobs"]),) 
                    for result in batch_results
                ])
            else:
                raise Exception(f"Error from server: {response.text}")
                
        return results
        
    def tokenize(self, text):
        response = requests.post(
            f"{self.server_url}/tokenize",
            json={"text": text}
        )
        
        if response.status_code == 200:
            return response.json()["tokens"]
        else:
            raise Exception(f"Error from server: {response.text}")

    def detokenize(self, tokens):
        response = requests.post(
            f"{self.server_url}/detokenize",
            json={"tokens": tokens}
        )
        
        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise Exception(f"Error from server: {response.text}")
    
    @property
    def eot_token_id(self):
        response = requests.get(f"{self.server_url}/model_info")
        if response.status_code == 200:
            return response.json()["eot_token_id"]
        else:
            raise Exception(f"Error from server: {response.text}")
    
    @property
    def max_length(self):
        response = requests.get(f"{self.server_url}/model_info")
        if response.status_code == 200:
            return response.json()["max_length"]
        else:
            raise Exception(f"Error from server: {response.text}")