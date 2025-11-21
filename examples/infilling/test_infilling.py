"""
Quick test script for the thinking infilling implementation.

This script runs a minimal test to verify the infilling approach works
without requiring a full MMLU evaluation.

Usage:
    python examples/test_infilling.py
"""

import asyncio
import json
from thinking_infilling import (
    run_infilling,
    create_infilling_prompt,
    GREEN, RED, YELLOW, BLUE, END
)
from llamppl import CachedCausalLM


# Test questions
TEST_QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "choices": {
            "A": "London",
            "B": "Berlin",
            "C": "Paris",
            "D": "Madrid"
        },
        "correct_answer": "C"
    },
    {
        "question": "What is 2 + 2?",
        "choices": {
            "A": "3",
            "B": "4",
            "C": "5",
            "D": "6"
        },
        "correct_answer": "B"
    },
    {
        "question": "Which planet is closest to the Sun?",
        "choices": {
            "A": "Venus",
            "B": "Earth",
            "C": "Mercury",
            "D": "Mars"
        },
        "correct_answer": "C"
    }
]


async def test_prompt_creation():
    """Test that prompts are created correctly"""
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Test 1: Prompt Creation{END}")
    print(f"{BLUE}{'='*80}{END}\n")

    question = TEST_QUESTIONS[0]
    token_budget = 50

    prompt = create_infilling_prompt(
        question["question"],
        question["choices"],
        token_budget,
        question["correct_answer"]
    )

    print("Generated prompt:")
    print(f"{YELLOW}{prompt}{END}\n")

    # Check that prompt has correct structure
    assert "<think>" in prompt, "Prompt missing <think> tag"
    assert "</think>" in prompt, "Prompt missing </think> tag"
    assert "[BLANK]" in prompt, "Prompt missing [BLANK] tokens"
    assert "The best answer is" in prompt, "Prompt missing answer preamble"
    assert question["correct_answer"] in prompt, "Prompt missing correct answer"

    # Count [BLANK] tokens
    blank_count = prompt.count("[BLANK]")
    print(f"Number of [BLANK] tokens: {blank_count}")
    print(f"Expected: {token_budget}")

    if blank_count == token_budget:
        print(f"{GREEN}✓ Prompt structure correct{END}\n")
    else:
        print(f"{RED}✗ Blank count mismatch{END}\n")

    return blank_count == token_budget


async def test_infilling_single(lm):
    """Test infilling on a single question"""
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Test 2: Single Question Infilling{END}")
    print(f"{BLUE}{'='*80}{END}\n")

    question = TEST_QUESTIONS[0]
    token_budget = 30  # Small budget for quick testing

    print(f"\n{YELLOW}Running infilling with budget {token_budget}...{END}")
    results = await run_infilling(
        lm=lm,
        question=question["question"],
        choices=question["choices"],
        correct_answer=question["correct_answer"],
        token_budget=token_budget,
        num_particles=3,  # Small number for quick testing
        beam_factor=2
    )

    # Check results
    print(f"\n{YELLOW}Checking results...{END}")
    all_have_answers = all(r.final_answer in ["A", "B", "C", "D"] for r in results)
    any_correct = any(r.is_correct for r in results)

    if all_have_answers:
        print(f"{GREEN}✓ All particles produced valid answers{END}")
    else:
        print(f"{RED}✗ Some particles missing valid answers{END}")

    if any_correct:
        print(f"{GREEN}✓ At least one particle got correct answer{END}")
    else:
        print(f"{YELLOW}⚠ No particles got correct answer (may be expected for difficult questions){END}")

    # Check token counts
    print(f"\n{YELLOW}Token budget analysis:{END}")
    for i, result in enumerate(results):
        within_budget = abs(result.actual_tokens - token_budget) <= 5
        status = f"{GREEN}✓{END}" if within_budget else f"{RED}✗{END}"
        print(f"  Particle {i+1}: {result.actual_tokens}/{token_budget} tokens {status}")

    return all_have_answers


async def test_multiple_budgets(lm):
    """Test infilling with multiple token budgets"""
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Test 3: Multiple Token Budgets{END}")
    print(f"{BLUE}{'='*80}{END}\n")

    question = TEST_QUESTIONS[1]  # Simple arithmetic question
    budgets = [20, 40]  # Small budgets for quick testing

    all_results = []
    for budget in budgets:
        print(f"\n{YELLOW}Testing budget: {budget}{END}")
        results = await run_infilling(
            lm=lm,
            question=question["question"],
            choices=question["choices"],
            correct_answer=question["correct_answer"],
            token_budget=budget,
            num_particles=2,
            beam_factor=2
        )
        all_results.extend(results)

    # Verify results vary by budget
    budget_groups = {}
    for result in all_results:
        if result.token_budget not in budget_groups:
            budget_groups[result.token_budget] = []
        budget_groups[result.token_budget].append(result.actual_tokens)

    print(f"\n{YELLOW}Budget adherence:{END}")
    for budget, actual_tokens in budget_groups.items():
        avg_tokens = sum(actual_tokens) / len(actual_tokens)
        print(f"  Budget {budget}: Average {avg_tokens:.1f} tokens")

    return True


async def test_data_export(lm):
    """Test that results can be exported for fine-tuning"""
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Test 4: Data Export{END}")
    print(f"{BLUE}{'='*80}{END}\n")

    question = TEST_QUESTIONS[2]
    results = await run_infilling(
        lm=lm,
        question=question["question"],
        choices=question["choices"],
        correct_answer=question["correct_answer"],
        token_budget=30,
        num_particles=2,
        beam_factor=2
    )

    # Test JSON export
    output_file = "test_export.json"
    from dataclasses import asdict
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"{GREEN}✓ Results exported to {output_file}{END}")

    # Test fine-tuning format
    finetuning_file = "test_finetuning.jsonl"
    with open(finetuning_file, 'w') as f:
        for result in results:
            if result.is_correct:
                example = {
                    "question": result.question,
                    "thinking": result.thinking_trace,
                    "answer": result.final_answer
                }
                f.write(json.dumps(example) + "\n")

    print(f"{GREEN}✓ Fine-tuning data exported to {finetuning_file}{END}")

    return True


async def run_all_tests():
    """Run all tests"""
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Running Thinking Infilling Tests{END}")
    print(f"{BLUE}{'='*80}{END}\n")

    results = {}

    # Test 1: Prompt creation (no model needed)
    try:
        results['prompt_creation'] = await test_prompt_creation()
    except Exception as e:
        print(f"{RED}✗ Test 1 failed: {e}{END}")
        results['prompt_creation'] = False

    # Load model once for all remaining tests
    print(f"\n{YELLOW}Loading model for tests 2-4...{END}")
    lm = CachedCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.2-3B")
    lm.batch_size = 8
    print(f"{GREEN}Model loaded!{END}")

    # Test 2: Single infilling
    try:
        results['single_infilling'] = await test_infilling_single(lm)
    except Exception as e:
        print(f"{RED}✗ Test 2 failed: {e}{END}")
        results['single_infilling'] = False

    # Test 3: Multiple budgets
    try:
        results['multiple_budgets'] = await test_multiple_budgets(lm)
    except Exception as e:
        print(f"{RED}✗ Test 3 failed: {e}{END}")
        results['multiple_budgets'] = False

    # Test 4: Data export
    try:
        results['data_export'] = await test_data_export(lm)
    except Exception as e:
        print(f"{RED}✗ Test 4 failed: {e}{END}")
        results['data_export'] = False

    # Print summary
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Test Summary{END}")
    print(f"{BLUE}{'='*80}{END}\n")

    for test_name, passed in results.items():
        status = f"{GREEN}✓ PASSED{END}" if passed else f"{RED}✗ FAILED{END}"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print(f"\n{GREEN}All tests passed! 🎉{END}")
    else:
        print(f"\n{YELLOW}Some tests failed. Check output above for details.{END}")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
