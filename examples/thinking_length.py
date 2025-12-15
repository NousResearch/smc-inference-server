from llamppl import Model, CachedCausalLM, LMContext, smc_steer
from typing import List
import asyncio

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
END = "\033[0m"

NUM_PARTICLES = 3
BEAM_FACTOR = 1
THINKING_TOKENS = 20
ITERATIONS = 1

PROMPT = "What is 2+2? <think>"

def pretty_format(particle):
    """This method formats particle contexts for display"""
    context_str = str(particle.context)
    return f"{context_str} (weight: {RED}{particle.weight:.3f}{END})"

class FixedLengthThinkingModel(Model):
    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        num_tokens: int = 10,
        temperature: float = 1.0
    ):
        super().__init__()
        self.lm = lm
        self.context = LMContext(lm, prompt, temperature)
        self.num_tokens = num_tokens
        self.generated_tokens = []
        self.is_reasoning = True
        self.eos_token = self.lm.tokenizer.eos_token_id

        self.period_tokens = set()
        for i, token in enumerate(self.lm.vocab):
            if token.endswith('.'):
                self.period_tokens.add(i)

    async def step(self):
        current_length = len(self.generated_tokens)

        # The model is currently reasoning
        if self.is_reasoning:
            if current_length >= self.num_tokens:
                self.condition(current_length == self.num_tokens)
                self.condition(self.generated_tokens[-1].token_id in self.period_tokens)
                await self.observe(self.context.next_token(), self.lm.vocab.index("\n"))
                for token_id in self.lm.tokenizer.encode("</think>"):
                    await self.observe(self.context.next_token(), token_id)
                for token_id in self.lm.tokenizer.encode("The best answer is"):
                    await self.observe(self.context.next_token(), token_id)
                self.is_reasoning = False

            if current_length == self.num_tokens - 1:
                period_mask = self.period_tokens
                await self.observe(self.context.mask_dist(period_mask), True) # Observe only EOS tokens (periods)
            elif current_length == self.num_tokens - 2:
                non_period_mask = set(range(len(self.lm.vocab))) - self.period_tokens - {self.lm.tokenizer.eos_token_id}
                await self.observe(self.context.mask_dist(non_period_mask), True) # Observe only non-EOS tokens
            else:
                await self.observe(self.context.mask_dist(set(self.lm.tokenizer.encode("</think>"))), False) # Observe any token but </think>

        # Normal sampling behavior
        next_dist = self.context.next_token()
        token = await self.sample(next_dist)
        self.generated_tokens.append(token)

        if token.token_id == self.lm.tokenizer.eos_token_id or len(self.generated_tokens) > (self.num_tokens + 50):
            self.finish()

    def get_result(self) -> str:
        """Return the generated text."""
        return self.lm.tokenizer.decode([t.token_id for t in self.generated_tokens])

async def main():
    lm = CachedCausalLM.from_pretrained("NousResearch/DeepHermes-3-Llama-3-3B-Preview")
    lm.batch_size = 8

    model = FixedLengthThinkingModel(
        lm=lm,
        prompt=PROMPT,
        num_tokens=THINKING_TOKENS,
        temperature=1.0
    )

    print(f"\nSteering with smc_steer with {GREEN}{NUM_PARTICLES}{END} particles and beam factor {GREEN}{BEAM_FACTOR}{END}")

    particles = await smc_steer(model, NUM_PARTICLES, BEAM_FACTOR)

    print(f"\n{GREEN}{PROMPT}{END}")

    for particle in particles:
        print(pretty_format(particle))
        print("Number of generated tokens: {y}{tokens}{e}".format(y=YELLOW, tokens=len(lm.tokenizer.encode(str(particle.context)))-1, e=END) + "\n")

if __name__ == "__main__":
    asyncio.run(main())