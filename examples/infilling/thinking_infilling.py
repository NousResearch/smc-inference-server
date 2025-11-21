"""
Thinking Trace Infilling Example with SMC

This example demonstrates how to use SMC for infilling reasoning traces.
Given a prompt with [BLANK] tokens between <think> and </think> tags,
the model generates a coherent thinking trace that leads to a correct answer
within a specified token budget.

This is particularly useful for:
- Generating training data for reasoning models
- Controlling the length of reasoning traces
- Ensuring reasoning traces lead to correct answers
"""

import asyncio
import json
import re
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from llamppl import Model, CachedCausalLM, LMContext, smc_steer, Token

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
END = "\033[0m"

# Configuration
NUM_PARTICLES = 3  # Reduced from 5 for speed
BEAM_FACTOR = 1    # Reduced from 2 for speed (less resampling)
TEMPERATURE = 1.0


@dataclass
class InfillingResult:
    """Stores the result of a single infilling attempt"""
    question: str
    correct_answer: str
    token_budget: int
    thinking_trace: str
    final_answer: str
    is_correct: bool
    actual_tokens: int
    weight: float
    timestamp: str


def pretty_format(particle, token_budget: int):
    """Format particle contexts for display with token counting"""
    context_str = str(particle.context)

    # Highlight the thinking section
    highlighted = re.sub(
        r'(<think>.*?</think>)',
        f'{YELLOW}\\1{END}',
        context_str,
        flags=re.DOTALL
    )

    # Highlight the answer
    highlighted = re.sub(
        r'(The best answer is [A-D])',
        f'{GREEN}\\1{END}',
        highlighted
    )

    return f"{highlighted}\n{BLUE}Weight: {particle.weight:.4f}{END}\n"


def create_infilling_prompt(question: str, choices: Dict[str, str], token_budget: int, correct_answer: str) -> str:
    """
    Creates a prompt with [BLANK] tokens for infilling.

    Args:
        question: The question to answer
        choices: Dict mapping choice letters (A, B, C, D) to choice text
        token_budget: Number of tokens allocated for thinking
        correct_answer: The correct answer (A, B, C, or D) - included in prompt

    Returns:
        Formatted prompt with blanks
    """
    # Format choices
    choices_text = "\n".join([f"{k}. {v}" for k, v in choices.items()])

    # Create the blank tokens
    blanks = " ".join(["[BLANK]"] * token_budget)

    prompt = f"""Question: {question}

{choices_text}

<think>
{blanks}
</think>

The best answer is {correct_answer}"""

    return prompt


class ThinkingInfillingModel(Model):
    """
    SMC model for infilling thinking traces.

    Uses 1 generated token per [BLANK] token for fast, controlled generation.
    """

    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        token_budget: int,
        temperature: float = 1.0
    ):
        super().__init__()
        self.lm = lm
        self.temperature = temperature

        # Split prompt on [BLANK] - each blank gets 1 generated token
        self.parts = prompt.split("[BLANK]")

        # Initialize context with first part
        self.context = LMContext(lm, self.parts[0], temperature)

        # Track state
        self.current_part_idx = 1
        self.thinking_tokens = []

    async def step(self):
        """Generate 1 token per blank, then observe the next part"""

        if self.current_part_idx >= len(self.parts):
            self.finish()
            return

        # Generate 1 token for this blank
        next_dist = self.context.next_token()
        token = await self.sample(next_dist)
        self.thinking_tokens.append(token)

        # Observe the next part after the blank
        part = self.parts[self.current_part_idx]
        part_tokens = self.lm.tokenizer.encode(part, add_special_tokens=False)

        for token_id in part_tokens:
            t = Token(self.lm, token_id, self.lm.tokenizer.decode([token_id]))
            await self.observe(self.context.next_token(), t)

        self.current_part_idx += 1

    def get_thinking_trace(self) -> str:
        """Extract just the thinking trace"""
        return self.lm.tokenizer.decode([t.token_id for t in self.thinking_tokens])

    def get_full_output(self) -> str:
        """Get the full generated output"""
        return str(self.context)


async def run_infilling(
    lm: CachedCausalLM,
    question: str,
    choices: Dict[str, str],
    correct_answer: str,
    token_budget: int,
    num_particles: int = NUM_PARTICLES,
    beam_factor: int = BEAM_FACTOR
) -> List[InfillingResult]:
    """
    Run SMC infilling for a single question.

    Returns a list of InfillingResult objects, one per particle.
    """
    prompt = create_infilling_prompt(question, choices, token_budget, correct_answer)

    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Question:{END} {question}")
    print(f"{GREEN}Correct Answer:{END} {correct_answer}")
    print(f"{GREEN}Token Budget:{END} {token_budget}")
    print(f"{YELLOW}Running SMC with {num_particles} particles...{END}")
    print(f"{BLUE}{'='*80}{END}\n")

    model = ThinkingInfillingModel(
        lm=lm,
        prompt=prompt,
        token_budget=token_budget,
        temperature=TEMPERATURE
    )

    import time
    start = time.time()
    particles = await smc_steer(model, num_particles, beam_factor)
    elapsed = time.time() - start
    print(f"\n{GREEN}✓ SMC completed in {elapsed:.1f}s{END}")

    results = []
    for i, particle in enumerate(particles):
        print(f"\n{YELLOW}Particle {i+1}:{END}")
        print(pretty_format(particle, token_budget))

        # Extract thinking and answer
        full_text = str(particle.context)
        thinking_match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
        thinking_trace = thinking_match.group(1).strip() if thinking_match else ""

        answer_match = re.search(r'The best answer is ([A-D])', full_text)
        final_answer = answer_match.group(1) if answer_match else ""

        # Count actual tokens in thinking trace
        thinking_tokens = lm.tokenizer.encode(thinking_trace, add_special_tokens=False)
        actual_tokens = len(thinking_tokens)

        result = InfillingResult(
            question=question,
            correct_answer=correct_answer,
            token_budget=token_budget,
            thinking_trace=thinking_trace,
            final_answer=final_answer,
            is_correct=(final_answer == correct_answer),
            actual_tokens=actual_tokens,
            weight=particle.weight,
            timestamp=datetime.now().isoformat()
        )
        results.append(result)

        # Print summary
        correct_marker = f"{GREEN}✓{END}" if result.is_correct else f"{RED}✗{END}"
        print(f"{correct_marker} Answer: {final_answer} (Correct: {correct_answer})")
        print(f"Tokens used: {actual_tokens}/{token_budget}")

    return results


async def main_single_example():
    """Run a single example to demonstrate the approach"""

    # Load model
    print(f"{YELLOW}Loading model...{END}")
    lm = CachedCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.2-3B")
    lm.batch_size = 16

    # Example question
    question = "What is the primary function of mitochondria in cells?"
    choices = {
        "A": "Protein synthesis",
        "B": "Energy production",
        "C": "DNA replication",
        "D": "Cell division"
    }
    correct_answer = "B"

    # Try different token budgets
    token_budgets = [50, 100, 150]

    all_results = []
    for budget in token_budgets:
        results = await run_infilling(
            lm=lm,
            question=question,
            choices=choices,
            correct_answer=correct_answer,
            token_budget=budget,
            num_particles=NUM_PARTICLES,
            beam_factor=BEAM_FACTOR
        )
        all_results.extend(results)

    # Save results
    output_file = "infilling_results_example.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    print(f"\n{GREEN}Results saved to {output_file}{END}")

    # Print summary statistics
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Summary Statistics:{END}")
    for budget in token_budgets:
        budget_results = [r for r in all_results if r.token_budget == budget]
        num_correct = sum(1 for r in budget_results if r.is_correct)
        print(f"  Budget {budget}: {num_correct}/{len(budget_results)} correct")
    print(f"{BLUE}{'='*80}{END}")


if __name__ == "__main__":
    asyncio.run(main_single_example())
