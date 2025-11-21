#!/bin/bash

# Helper script to run the thinking infilling examples
# Usage: ./run_infilling_examples.sh [test|example|eval|full]

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Get the command
COMMAND=${1:-help}

case $COMMAND in
    test)
        print_header "Running Tests"
        print_info "This will verify the infilling implementation works correctly"
        python examples/test_infilling.py
        ;;

    example)
        print_header "Running Basic Example"
        print_info "This will generate thinking traces for a single question"
        python examples/thinking_infilling.py
        ;;

    eval)
        print_header "Running Small MMLU Evaluation"
        print_info "Evaluating on 10 MMLU questions with 2 token budgets"
        python examples/mmlu_infilling_eval.py \
            --num_questions 10 \
            --token_budgets 50,100 \
            --num_particles 3 \
            --output_dir mmlu_test_results

        if [ $? -eq 0 ]; then
            print_info "Results saved to mmlu_test_results/"
            print_info "Check mmlu_test_results/statistics_*.json for results"
        fi
        ;;

    full)
        print_header "Running Full MMLU Evaluation"
        print_info "Evaluating on 100 MMLU questions with 4 token budgets"
        print_info "This will take 30-60 minutes..."

        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python examples/mmlu_infilling_eval.py \
                --num_questions 100 \
                --token_budgets 100,250,500,1000 \
                --num_particles 5 \
                --beam_factor 2 \
                --output_dir mmlu_full_results \
                --export_finetuning

            if [ $? -eq 0 ]; then
                print_info "Results saved to mmlu_full_results/"
                print_info "Fine-tuning data: mmlu_full_results/finetuning_data.jsonl"
            fi
        fi
        ;;

    finetuning)
        print_header "Generating Fine-tuning Dataset"
        print_info "Generating 500 examples for fine-tuning"

        python examples/mmlu_infilling_eval.py \
            --num_questions 500 \
            --token_budgets 100,250,500,1000 \
            --num_particles 10 \
            --beam_factor 3 \
            --output_dir finetuning_dataset \
            --export_finetuning

        if [ $? -eq 0 ]; then
            print_info "Dataset saved to finetuning_dataset/"

            # Count examples
            if [ -f "finetuning_dataset/finetuning_data.jsonl" ]; then
                num_examples=$(wc -l < finetuning_dataset/finetuning_data.jsonl)
                print_info "Generated ${num_examples} fine-tuning examples"
            fi
        fi
        ;;

    clean)
        print_header "Cleaning Up Test Files"
        rm -f test_export.json test_finetuning.jsonl infilling_results_example.json
        rm -rf mmlu_test_results/
        print_info "Cleaned up test files"
        ;;

    help|*)
        echo "Thinking Infilling Examples Runner"
        echo ""
        echo "Usage: ./run_infilling_examples.sh [command]"
        echo ""
        echo "Commands:"
        echo "  test         - Run unit tests to verify implementation"
        echo "  example      - Run basic example with single question"
        echo "  eval         - Run small MMLU eval (10 questions, quick)"
        echo "  full         - Run full MMLU eval (100 questions, ~1 hour)"
        echo "  finetuning   - Generate large fine-tuning dataset (500 questions)"
        echo "  clean        - Remove test files and temporary results"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_infilling_examples.sh test"
        echo "  ./run_infilling_examples.sh example"
        echo "  ./run_infilling_examples.sh eval"
        ;;
esac
