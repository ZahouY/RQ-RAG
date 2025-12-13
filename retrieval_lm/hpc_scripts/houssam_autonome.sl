#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=24G
#SBATCH --account=m25206
#SBATCH --job-name=rqrag



source ~/rqrag/bin/activate

cd retrieval_lm

export HUGGINGFACEHUB_API_TOKEN="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"

# Exemple 1 : question unique passée en argument
python ./houssam_autonome.py \
  --question "In which country is the university where Michelle Obama studied law located?" \
  --max_steps 4 \
  --max_new_tokens_step 128 \
  --max_web_results 3

# Exemple 2 (en commentaire) : lire un fichier de questions
# python rqrag_agent_autonome.py \
#   --questions_file questions.txt \
#   --max_steps 4 \
#   --max_new_tokens_step 128 \
#   --max_web_results 3