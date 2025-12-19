#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --account=m25206
#SBATCH --job-name=rqrag

source ~/rqrag/bin/activate

cd retrieval_lm


# export PYTHONPATH="$(pwd):$PYTHONPATH"
# --pruning_early_stopping necessite --selection_strategy majority_vote

python ./inference.py \
--model_name_or_path \
"../models/rq_rag_llama2_7B" \
--input_file \
"data/hotpotqa_test.json" \
--output_path \
"../output/inference" \
--ndocs 3 \
--use_search_engine \
--use_hf \
--task popqa_longtail_w_gs \
--tree_decode \
--max_depth 2 \
--search_engine_type duckduckgo \
--expand_on_tokens \
"[S_Rewritten_Query]" \
"[S_Decomposed_Query]" \
"[S_Disambiguated_Query]" \
"[A_Response]" \
--max_new_tokens 128 \
--selection_strategy confidence_score
#--pruning_sanity_check \
#--pruning_early_stopping \
#--early_stopping_threshold 3