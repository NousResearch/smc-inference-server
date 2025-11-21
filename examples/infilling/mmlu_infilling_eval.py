"""
MMLU Evaluation with Thinking Trace Infilling

This script evaluates the thinking infilling approach on MMLU questions,
generating reasoning traces of different lengths that lead to correct answers.
The generated data can be used for fine-tuning reasoning models.

Usage:
    python mmlu_infilling_eval.py --num_questions 100 --token_budgets 100,250,500,1000

Requirements:
    pip install datasets
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

from datasets import load_dataset
from llamppl import CachedCausalLM

# Import from the infilling example
import sys
sys.path.append(str(Path(__file__).parent))
from thinking_infilling import (
    run_infilling,
    InfillingResult,
    NUM_PARTICLES,
    BEAM_FACTOR,
    GREEN, RED, YELLOW, BLUE, END
)


@dataclass
class MMLUQuestion:
    """Represents a single MMLU question"""
    question: str
    choices: Dict[str, str]
    correct_answer: str
    subject: str
    index: int


def load_mmlu_questions(
    num_questions: int = 100,
    subjects: Optional[List[str]] = None,
    split: str = "validation"
) -> List[MMLUQuestion]:
    """
    Load MMLU questions from HuggingFace datasets.

    Args:
        num_questions: Maximum number of questions to load
        subjects: List of MMLU subjects to include (None = all subjects)
        split: Dataset split to use ('validation', 'test', 'dev')

    Returns:
        List of MMLUQuestion objects
    """
    print(f"{YELLOW}Loading MMLU dataset...{END}")

    # Load the MMLU dataset
    # Using the 'cais/mmlu' dataset which is the standard one
    dataset = load_dataset("cais/mmlu", "all", split=split)

    questions = []
    for i, item in enumerate(dataset):
        # Skip if we have enough questions
        if len(questions) >= num_questions:
            break

        # Filter by subject if specified
        if subjects is not None and item['subject'] not in subjects:
            continue

        # Parse the question
        question_text = item['question']
        choices_list = item['choices']
        correct_idx = item['answer']

        # Create choices dict
        choice_letters = ['A', 'B', 'C', 'D']
        choices = {
            letter: choice
            for letter, choice in zip(choice_letters, choices_list)
        }
        correct_answer = choice_letters[correct_idx]

        questions.append(MMLUQuestion(
            question=question_text,
            choices=choices,
            correct_answer=correct_answer,
            subject=item['subject'],
            index=i
        ))

    print(f"{GREEN}Loaded {len(questions)} questions{END}")
    return questions


async def evaluate_mmlu_with_infilling(
    questions: List[MMLUQuestion],
    token_budgets: List[int],
    model_name: str = "NousResearch/Hermes-3-Llama-3.2-3B",
    num_particles: int = NUM_PARTICLES,
    beam_factor: int = BEAM_FACTOR,
    output_dir: str = "mmlu_infilling_results",
    save_frequency: int = 10
) -> List[InfillingResult]:
    """
    Evaluate MMLU questions with thinking trace infilling.

    Args:
        questions: List of MMLU questions to evaluate
        token_budgets: List of token budgets to try for each question
        model_name: HuggingFace model name
        num_particles: Number of SMC particles
        beam_factor: SMC beam factor
        output_dir: Directory to save results
        save_frequency: Save intermediate results every N questions

    Returns:
        List of all InfillingResult objects
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load model
    print(f"{YELLOW}Loading model: {model_name}{END}")
    lm = CachedCausalLM.from_pretrained(model_name)
    lm.batch_size = 16

    all_results = []
    start_time = datetime.now()

    # Process each question
    for q_idx, question in enumerate(tqdm(questions, desc="Processing questions")):
        question_results = []

        # Try each token budget
        for budget in token_budgets:
            try:
                results = await run_infilling(
                    lm=lm,
                    question=question.question,
                    choices=question.choices,
                    correct_answer=question.correct_answer,
                    token_budget=budget,
                    num_particles=num_particles,
                    beam_factor=beam_factor
                )

                # Add metadata
                for result in results:
                    result_dict = asdict(result)
                    result_dict['subject'] = question.subject
                    result_dict['question_index'] = question.index
                    question_results.append(result_dict)

                all_results.extend(question_results)

            except Exception as e:
                print(f"\n{RED}Error processing question {q_idx} with budget {budget}: {e}{END}")
                continue

        # Save intermediate results
        if (q_idx + 1) % save_frequency == 0:
            intermediate_file = output_path / f"intermediate_results_{q_idx+1}.json"
            with open(intermediate_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n{GREEN}Saved intermediate results to {intermediate_file}{END}")

    # Save final results
    final_file = output_path / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Calculate and save statistics
    stats = calculate_statistics(all_results, token_budgets)
    stats_file = output_path / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Evaluation Complete!{END}")
    print(f"Total questions: {len(questions)}")
    print(f"Total results: {len(all_results)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Results saved to: {final_file}")
    print(f"Statistics saved to: {stats_file}")
    print(f"{BLUE}{'='*80}{END}")

    print_statistics_summary(stats)

    return all_results


def calculate_statistics(results: List[Dict], token_budgets: List[int]) -> Dict:
    """Calculate statistics from results"""
    stats = {
        'overall': {
            'total_results': len(results),
            'num_correct': sum(1 for r in results if r['is_correct']),
            'accuracy': sum(1 for r in results if r['is_correct']) / len(results) if results else 0
        },
        'by_budget': {},
        'by_subject': {}
    }

    # Statistics by token budget
    for budget in token_budgets:
        budget_results = [r for r in results if r['token_budget'] == budget]
        if budget_results:
            stats['by_budget'][budget] = {
                'num_results': len(budget_results),
                'num_correct': sum(1 for r in budget_results if r['is_correct']),
                'accuracy': sum(1 for r in budget_results if r['is_correct']) / len(budget_results),
                'avg_actual_tokens': sum(r['actual_tokens'] for r in budget_results) / len(budget_results),
                'avg_weight': sum(r['weight'] for r in budget_results) / len(budget_results)
            }

    # Statistics by subject
    subjects = set(r['subject'] for r in results if 'subject' in r)
    for subject in subjects:
        subject_results = [r for r in results if r.get('subject') == subject]
        if subject_results:
            stats['by_subject'][subject] = {
                'num_results': len(subject_results),
                'num_correct': sum(1 for r in subject_results if r['is_correct']),
                'accuracy': sum(1 for r in subject_results if r['is_correct']) / len(subject_results)
            }

    return stats


def print_statistics_summary(stats: Dict):
    """Print a formatted summary of statistics"""
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{GREEN}Statistics Summary:{END}\n")

    # Overall stats
    overall = stats['overall']
    print(f"Overall Accuracy: {overall['accuracy']:.2%} ({overall['num_correct']}/{overall['total_results']})")

    # By token budget
    print(f"\n{YELLOW}By Token Budget:{END}")
    for budget, budget_stats in sorted(stats['by_budget'].items()):
        print(f"  {budget} tokens: {budget_stats['accuracy']:.2%} "
              f"({budget_stats['num_correct']}/{budget_stats['num_results']}) "
              f"[Avg actual: {budget_stats['avg_actual_tokens']:.1f}]")

    # By subject (top 10)
    if stats['by_subject']:
        print(f"\n{YELLOW}By Subject (Top 10 by accuracy):{END}")
        sorted_subjects = sorted(
            stats['by_subject'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )[:10]
        for subject, subject_stats in sorted_subjects:
            print(f"  {subject}: {subject_stats['accuracy']:.2%} "
                  f"({subject_stats['num_correct']}/{subject_stats['num_results']})")

    print(f"{BLUE}{'='*80}{END}\n")


def export_for_finetuning(
    results: List[Dict],
    output_file: str = "finetuning_data.jsonl",
    min_weight: float = 0.1,
    only_correct: bool = True
):
    """
    Export results in a format suitable for fine-tuning.

    Args:
        results: List of result dictionaries
        output_file: Output file path
        min_weight: Minimum particle weight to include
        only_correct: Only include results with correct answers
    """
    filtered_results = [
        r for r in results
        if r['weight'] >= min_weight and (not only_correct or r['is_correct'])
    ]

    with open(output_file, 'w') as f:
        for result in filtered_results:
            # Format as a training example
            choices_text = "\n".join([
                f"{k}. {v}"
                for k, v in result.get('choices', {}).items()
            ])

            training_example = {
                "question": result['question'],
                "choices": choices_text,
                "thinking": result['thinking_trace'],
                "answer": result['final_answer'],
                "metadata": {
                    "token_budget": result['token_budget'],
                    "actual_tokens": result['actual_tokens'],
                    "weight": result['weight'],
                    "subject": result.get('subject', 'unknown')
                }
            }

            f.write(json.dumps(training_example) + "\n")

    print(f"{GREEN}Exported {len(filtered_results)} examples to {output_file}{END}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU with thinking trace infilling"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=100,
        help="Number of MMLU questions to evaluate"
    )
    parser.add_argument(
        "--token_budgets",
        type=str,
        default="100,250,500,1000",
        help="Comma-separated list of token budgets to try"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of MMLU subjects to include (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NousResearch/Hermes-3-Llama-3.2-3B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=NUM_PARTICLES,
        help="Number of SMC particles"
    )
    parser.add_argument(
        "--beam_factor",
        type=int,
        default=BEAM_FACTOR,
        help="SMC beam factor"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mmlu_infilling_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test", "dev"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--export_finetuning",
        action="store_true",
        help="Export results for fine-tuning"
    )

    args = parser.parse_args()

    # Parse token budgets
    token_budgets = [int(x.strip()) for x in args.token_budgets.split(",")]

    # Parse subjects if provided
    subjects = None
    if args.subjects:
        subjects = [x.strip() for x in args.subjects.split(",")]

    # Load questions
    questions = load_mmlu_questions(
        num_questions=args.num_questions,
        subjects=subjects,
        split=args.split
    )

    # Run evaluation
    results = asyncio.run(evaluate_mmlu_with_infilling(
        questions=questions,
        token_budgets=token_budgets,
        model_name=args.model,
        num_particles=args.num_particles,
        beam_factor=args.beam_factor,
        output_dir=args.output_dir
    ))

    # Export for fine-tuning if requested
    if args.export_finetuning:
        output_file = Path(args.output_dir) / "finetuning_data.jsonl"
        export_for_finetuning(results, str(output_file))


if __name__ == "__main__":
    main()
