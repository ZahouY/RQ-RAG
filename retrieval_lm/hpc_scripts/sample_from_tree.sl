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

python output/sammple_from_tree.py \
--model_name_or_path \
"../models/rq_rag_llama2_7B" \
--original_data \
"data/hotpotqa_test.json" \
--run_name \
"../output/houssam" \
--task popqa_longtail_w_gs \
--calc_depth \
1 \
2 \
3 \
--calc_width \
--expand_on_tokens \
"[S_Rewritten_Query]" \
"[S_Decomposed_Query]" \
"[S_Disambiguated_Query]" \
"[A_Response]" \
--calc_retrieval_performance
