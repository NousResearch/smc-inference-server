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
NUM_PARTICLES = 5
BEAM_FACTOR = 2
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


def create_infilling_prompt(question: str, choices: Dict[str, str], token_budget: int) -> str:
    """
    Creates a prompt with [BLANK] tokens for infilling.

    Args:
        question: The question to answer
        choices: Dict mapping choice letters (A, B, C, D) to choice text
        token_budget: Number of tokens allocated for thinking

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

The best answer is"""

    return prompt


class ThinkingInfillingModel(Model):
    """
    SMC model for infilling thinking traces.

    This model:
    1. Replaces [BLANK] tokens with actual reasoning
    2. Enforces a specific token budget for thinking
    3. Ensures the trace ends properly with </think> and an answer
    4. Can optionally constrain to produce a specific correct answer
    """

    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        token_budget: int,
        correct_answer: Optional[str] = None,
        temperature: float = 1.0
    ):
        super().__init__()
        self.lm = lm
        self.context = LMContext(lm, prompt, temperature)
        self.token_budget = token_budget
        self.correct_answer = correct_answer
        self.generated_tokens = []

        # Track state
        self.in_thinking = False
        self.thinking_complete = False
        self.thinking_tokens = []

        # Parse the prompt to find where blanks start
        self._find_blank_positions()

        # Get special tokens
        self.eos_token = self.lm.tokenizer.eos_token_id
        self.blank_token_ids = set(self.lm.tokenizer.encode("[BLANK]", add_special_tokens=False))
        self.think_close_ids = self.lm.tokenizer.encode("</think>", add_special_tokens=False)

        # Tokens that typically end sentences
        self.sentence_end_tokens = self._get_sentence_end_tokens()

    def _find_blank_positions(self):
        """Find where [BLANK] tokens are in the context"""
        prompt_text = str(self.context)
        think_start = prompt_text.find("<think>")
        think_end = prompt_text.find("</think>")

        if think_start != -1 and think_end != -1:
            self.in_thinking = True
            thinking_section = prompt_text[think_start + 7:think_end]
            self.num_blanks = thinking_section.count("[BLANK]")

    def _get_sentence_end_tokens(self) -> set:
        """Get token IDs that typically end sentences"""
        end_tokens = set()
        for token_text in [".", "!", "?"]:
            token_ids = self.lm.tokenizer.encode(token_text, add_special_tokens=False)
            for tid in token_ids:
                # Check if token ends with punctuation
                decoded = self.lm.tokenizer.decode([tid])
                if decoded.rstrip().endswith(token_text):
                    end_tokens.add(tid)
        return end_tokens

    async def step(self):
        """Single step of generation with infilling constraints"""

        # Check if we've hit generation limits
        if len(self.generated_tokens) > self.token_budget + 100:
            self.finish()
            return

        # Handle [BLANK] token replacement during thinking
        if self.in_thinking and not self.thinking_complete:
            # Check if we're at a [BLANK] token
            next_dist = self.context.next_token()

            # Get the next token from the original prompt
            peek_token = await self.sample(next_dist)

            # If it's a [BLANK] token, replace it with generated content
            if peek_token.token_id in self.blank_token_ids:
                # Generate replacement token
                replacement_dist = self.context.next_token()

                # Don't allow generating more [BLANK] tokens or </think> yet
                forbidden_tokens = self.blank_token_ids | set(self.think_close_ids)
                await self.observe(self.context.mask_dist(forbidden_tokens), False)

                # If we're near the end of token budget, prefer sentence-ending tokens
                if len(self.thinking_tokens) >= self.token_budget - 2:
                    await self.observe(
                        self.context.mask_dist(self.sentence_end_tokens),
                        True
                    )

                replacement_token = await self.sample(replacement_dist)
                self.thinking_tokens.append(replacement_token)
                self.generated_tokens.append(replacement_token)

                # Check if we've filled the budget
                if len(self.thinking_tokens) >= self.token_budget:
                    self.thinking_complete = True
                    self.in_thinking = False

                    # Force generation of </think>
                    for token_id in self.think_close_ids:
                        close_token = Token(self.lm, token_id, self.lm.tokenizer.decode([token_id]))
                        await self.observe(self.context.next_token(), close_token)
                        self.generated_tokens.append(close_token)

                    # Force generation of newline and "The best answer is"
                    answer_preamble = "\nThe best answer is"
                    for token_id in self.lm.tokenizer.encode(answer_preamble, add_special_tokens=False):
                        ans_token = Token(self.lm, token_id, self.lm.tokenizer.decode([token_id]))
                        await self.observe(self.context.next_token(), ans_token)
                        self.generated_tokens.append(ans_token)

                return
            else:
                # Not a blank token, continue normal generation
                self.generated_tokens.append(peek_token)

                # Check if we hit </think> naturally
                if peek_token.token_id in self.think_close_ids:
                    self.thinking_complete = True
                    self.in_thinking = False

                return

        # After thinking, generate the answer
        if self.thinking_complete:
            next_dist = self.context.next_token()

            # If we have a correct answer constraint, enforce it
            if self.correct_answer is not None and len(self.generated_tokens) - len(self.thinking_tokens) < 5:
                # Try to steer toward the correct answer
                answer_tokens = self.lm.tokenizer.encode(f" {self.correct_answer}", add_special_tokens=False)
                if len(self.generated_tokens) - len(self.thinking_tokens) - len(self.think_close_ids) < len(answer_tokens):
                    expected_token_idx = len(self.generated_tokens) - len(self.thinking_tokens) - len(self.think_close_ids)
                    if expected_token_idx < len(answer_tokens):
                        expected_token_id = answer_tokens[expected_token_idx]
                        expected_token = Token(
                            self.lm,
                            expected_token_id,
                            self.lm.tokenizer.decode([expected_token_id])
                        )
                        await self.observe(next_dist, expected_token)
                        self.generated_tokens.append(expected_token)
                        return

            token = await self.sample(next_dist)
            self.generated_tokens.append(token)

            if token.token_id == self.eos_token:
                self.finish()

            return

        # Normal generation (shouldn't typically reach here)
        next_dist = self.context.next_token()
        token = await self.sample(next_dist)
        self.generated_tokens.append(token)

        if token.token_id == self.eos_token:
            self.finish()

    def get_thinking_trace(self) -> str:
        """Extract just the thinking trace"""
        full_text = self.lm.tokenizer.decode([t.token_id for t in self.generated_tokens])
        match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def get_answer(self) -> str:
        """Extract the final answer"""
        full_text = self.lm.tokenizer.decode([t.token_id for t in self.generated_tokens])
        match = re.search(r'The best answer is ([A-D])', full_text)
        if match:
            return match.group(1)
        return ""


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
    prompt = create_infilling_prompt(question, choices, token_budget)

    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Question:{END} {question}")
    print(f"{GREEN}Correct Answer:{END} {correct_answer}")
    print(f"{GREEN}Token Budget:{END} {token_budget}")
    print(f"{BLUE}{'='*80}{END}\n")

    model = ThinkingInfillingModel(
        lm=lm,
        prompt=prompt,
        token_budget=token_budget,
        correct_answer=correct_answer,
        temperature=TEMPERATURE
    )

    particles = await smc_steer(model, num_particles, beam_factor)

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
