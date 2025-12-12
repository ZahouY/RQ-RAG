#!/bin/bash

# Script pour copier le repo vers le HPC
# Usage: ./sync_to_hpc.sh

echo "=== Copie du repo RQ-RAG vers le HPC ==="

scp -r /home/zahou/Bureau/DLA/RQ-RAG yaszahou@juliet.mesonet.fr:~/

echo "=== Copie terminée ==="
