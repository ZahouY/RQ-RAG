# Créer un dossier pour les modèles
mkdir -p models

# Télécharger le modèle (cela peut prendre du temps, ~13GB)
huggingface-cli download zorowin123/rq_rag_llama2_7B --local-dir models/rq_rag_llama2_7B --local-dir-use-symlinks False