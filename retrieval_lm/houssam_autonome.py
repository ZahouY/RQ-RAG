#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import torch
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

try:
    from duckduckgo_search import DDGS
    DUCK_AVAILABLE = True
except ImportError:
    DUCK_AVAILABLE = False
    DDGS = None


# ========================================
#  Critère d'arrêt pour la génération
# ========================================
class EOSStoppingCriteria(StoppingCriteria):
    """Arrête la génération quand [EOS] est produit."""
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Vérifie si les derniers tokens contiennent [EOS]
        last_token = input_ids[0, -1].item()
        return last_token == self.eos_token_id


# -----------------------------
#  Recherche web (DuckDuckGo)
# -----------------------------
def web_search(query: str, max_results: int = 3):
    """
    Retourne une liste de snippets textuels pour le RAG.
    Si DuckDuckGo n'est pas dispo, renvoie une liste vide.
    """
    if not DUCK_AVAILABLE:
        print("⚠️ duckduckgo_search non disponible, pas de recherche web.")
        return []

    results = []
    try:
        with DDGS(timeout=20) as ddgs:
            for r in ddgs.text(
                keywords=query,
                max_results=max_results,
                safesearch="moderate",
                region="wt-wt",
            ):
                results.append(r)
    except Exception as e:
        print(f"❌ Erreur DuckDuckGo pour la requête '{query}': {e}")
        return []

    snippets = []
    for r in results:
        title = r.get("title") or ""
        body = r.get("body") or r.get("description") or ""
        txt = (title + " - " + body).strip()
        if txt:
            snippets.append(txt)

    return snippets


def format_evidences(snippets):
    """
    Formate les snippets façon 'R_Evidences'.
    """
    if not snippets:
        return "Title: dummy\nText: no evidence retrieved\n"

    docs = []
    for i, s in enumerate(snippets):
        docs.append(f"Title: doc{i}\nText: {s}")
    return "\n\n".join(docs)


# -----------------------------
#  Prompt agent autonome
# -----------------------------
def build_agent_prompt(question: str) -> str:
    """
    Prompt de départ : explique au modèle les tokens d'action.
    """
    system_msg = """You are an RQ-RAG agent.

You can use the following actions in your answer:
- [S_Rewritten_Query] ... [EOS]       to rewrite the question as a search query
- [S_Decomposed_Query] ... [EOS]      to decompose the question into simpler subquestions
- [S_Disambiguated_Query] ... [EOS]   to disambiguate an unclear question
- [A_Response] ... [EOS]              to give the final answer

You are NOT allowed to use [A_Response] before you have used at least one [S...] action.

Each time you use an [S_...] token, you MUST:
1) Write ONLY the query or sub-question text.
2) Then output [EOS].

After that, external tools may return evidence to you inside:
[R_Evidences] ... [/R_Evidences]

When you are ready to answer, use:
[A_Response] final answer here [EOS]
"""

    user_msg = f"Question: {question}"

    prompt = (
        "<s><|system|>\n" + system_msg + "\n</s>\n"
        "<|user|>\n" + user_msg + "\n</s>\n"
        "<|assistant|>\n"
    )
    return prompt


# ========================================
#  Agent autonome RQ-RAG
# ========================================
def rqrag_agent_autonome(
    model,
    tokenizer,
    question: str,
    max_steps: int = 4,
    max_new_tokens_step: int = 200,
    max_web_results: int = 3,
):
    """
    Implémente un RQ-RAG 'agent autonome' :
    - le modèle décide autonomously quand émettre [S_...] ou [A_Response]
    - chaque [S_...] déclenche une recherche web
    - affiche les étapes détaillées avant la réponse finale
    """
    history_text = build_agent_prompt(question)
    actions_log = []
    seen_actions = set()
    full_conversation = []

    print("\n" + "="*50)
    print(f"🤖 QUESTION: {question}")
    print("="*50)

    for step in range(max_steps):
        print(f"\n{'─'*50}")
        print(f"📍 ÉTAPE {step+1}/{max_steps}")
        print(f"{'─'*50}")

        # Tokenize l'historique
        inputs = tokenizer(
            history_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        print("🔄 Génération en cours...")

        # Générer les tokens
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens_step,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=1,
            )

        # Extraire UNIQUEMENT les nouveaux tokens générés
        new_tokens = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # Afficher ce qui a été généré
        print(f"\n📝 Texte généré:\n{generated_text}")

        # Sauvegarder la génération complète
        full_conversation.append(generated_text)

        # ====== 1) Chercher [A_Response] =====
        ans_matches = re.findall(
            r"\[A_Response\](.*?)\[EOS\]",
            generated_text,
            flags=re.DOTALL
        )
        
        if ans_matches:
            final_answer = ans_matches[-1].strip()
            
            # Vérifier que le modèle a utilisé au moins une action [S_...]
            if len(actions_log) == 0:
                print("\n⚠️ ALERTE: [A_Response] détecté SANS actions préalables [S_...]")
                print("🔁 Rejet de la réponse. Le modèle doit d'abord faire une recherche/décomposition.")
                
                # Ajouter un feedback au modèle
                history_text = (
                    history_text + generated_text +
                    "\n<|system|>\n⚠️ ERROR: You must use at least one [S_...] action before [A_Response]!\n"
                    "Please start by rewriting the question or decomposing it.\n</s>\n"
                    "<|assistant|>\n"
                )
                continue
            
            print(f"\n✅ [A_Response] DÉTECTÉ après {len(actions_log)} action(s)")
            print(f"📌 RÉPONSE FINALE: {final_answer}\n")
            
            return {
                "answer": final_answer,
                "full_conversation": "\n".join(full_conversation),
                "actions": actions_log,
                "stopped_by": "A_Response",
                "num_steps": step + 1,
            }

        # ====== 2) Chercher les actions [S_...] =====
        act_matches = re.findall(
            r"\[(S_Rewritten_Query|S_Decomposed_Query|S_Disambiguated_Query)\](.*?)\[EOS\]",
            generated_text,
            flags=re.DOTALL,
        )

        if act_matches:
            last_action, last_query = act_matches[-1]
            last_query = last_query.strip()

            # Vérifier si on boucle
            if (last_action, last_query) in seen_actions:
                print(f"\n⚠️ BOUCLE DÉTECTÉE: Même action '{last_action}' avec même query")
                print("❌ Arrêt pour éviter une boucle infinie.")
                break

            seen_actions.add((last_action, last_query))

            # Afficher l'action
            action_display = {
                "S_Rewritten_Query": "� RÉÉCRITURE DE REQUÊTE",
                "S_Decomposed_Query": "🔀 DÉCOMPOSITION",
                "S_Disambiguated_Query": "❓ DÉSAMBIGUÏSATION",
            }
            
            print(f"\n{action_display.get(last_action, last_action)}")
            print(f"  → Requête: {last_query}")

            # Recherche web
            print("  🔍 Recherche web...")
            snippets = web_search(last_query, max_results=max_web_results)
            
            if snippets:
                print(f"  ✓ {len(snippets)} résultat(s) trouvé(s)")
            else:
                print("  ✗ Aucun résultat trouvé")

            ev_text = format_evidences(snippets)

            actions_log.append({
                "action": last_action,
                "query": last_query,
                "snippets": snippets,
            })

            # Réinjecter les évidences pour la prochaine itération
            history_text = (
                history_text + generated_text +
                "\n[R_Evidences]\n" +
                ev_text +
                "\n[/R_Evidences]\n" +
                "<|assistant|>\n"
            )
            continue

        # ====== 3) Rien détecté → on arrête =====
        print("\n⚠️ Aucun token spécial [S_...] ou [A_Response] détecté.")
        print("❌ Arrêt de la génération.")
        break

    # Fin sans réponse
    print("\n" + "="*50)
    print("❌ IMPOSSIBLE DE GÉNÉRER UNE RÉPONSE")
    print(f"(Arrêt après {len(actions_log)} action(s), {step+1} étape(s))")
    print("="*50 + "\n")
    
    return {
        "answer": None,
        "full_conversation": "\n".join(full_conversation),
        "actions": actions_log,
        "stopped_by": "max_steps_or_no_action",
        "num_steps": step + 1,
    }


# ========================================
#  Chargement du modèle
# ========================================
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    model_name = os.environ.get("RQRAG_MODEL_NAME", "zorowin123/rq_rag_llama2_7B")
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

    # Charger tokenizer et modèle
    tokenizer = AutoTokenizer.from_pretrained(
        "../models/rq_rag_llama2_7B",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "../models/rq_rag_llama2_7B",
        device_map="auto",
    )

    # Ajouter les tokens spéciaux au vocabulaire
    special_tokens = {
        "additional_special_tokens": [
            "[S_Rewritten_Query]",
            "[S_Decomposed_Query]",
            "[S_Disambiguated_Query]",
            "[A_Response]",
            "[R_Evidences]",
            "[/R_Evidences]",
            "[EOS]",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print("✅ Modèle chargé :", model_name)
    print(f"✅ Tokens spéciaux ajoutés. Vocabulaire: {len(tokenizer)} tokens")
    
    return model, tokenizer


# -----------------------------
#  Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test 'agent autonome' pour RQ-RAG avec génération autonome des tokens spéciaux."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question unique à tester.",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="Fichier texte avec une question par ligne.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=4,
        help="Nombre max d'étapes/actions [S_...] avant d'arrêter.",
    )
    parser.add_argument(
        "--max_new_tokens_step",
        type=int,
        default=200,
        help="Nombre max de tokens générés par étape (augmenté pour laisser place aux tokens spéciaux).",
    )
    parser.add_argument(
        "--max_web_results",
        type=int,
        default=3,
        help="Nombre max de résultats web par requête de recherche.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.question is None and args.questions_file is None:
        print("❌ Erreur: Spécifie soit --question, soit --questions_file.")
        return

    print("\n" + "="*60)
    print("🚀 RQ-RAG AGENT AUTONOME - DÉMARRAGE")
    print("="*60 + "\n")

    model, tokenizer = load_model_and_tokenizer()

    questions = []
    if args.question is not None:
        questions.append(args.question)

    if args.questions_file is not None:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(line)

    print(f"📋 {len(questions)} question(s) à traiter\n")

    results_summary = []

    for idx, q in enumerate(questions, 1):
        print(f"\n{'#'*60}")
        print(f"# Question {idx}/{len(questions)}")
        print(f"{'#'*60}")
        
        res = rqrag_agent_autonome(
            model,
            tokenizer,
            q,
            max_steps=args.max_steps,
            max_new_tokens_step=args.max_new_tokens_step,
            max_web_results=args.max_web_results,
        )
        
        # Afficher le récapitulatif pour cette question
        print("\n" + "─"*60)
        print("📊 RÉCAPITULATIF DE LA QUESTION")
        print("─"*60)
        print(f"\n❓ Question: {q}\n")
        
        print(f"📈 Statut: {res['stopped_by']}")
        print(f"⏱️ Étapes effectuées: {res['num_steps']}/{args.max_steps}")
        print(f"🔎 Actions exécutées: {len(res['actions'])}")
        
        if res['actions']:
            print("\n📍 Détail des actions:")
            for i, action in enumerate(res['actions'], 1):
                action_name = action['action'].replace('S_', '').replace('_', ' ')
                print(f"  {i}. [{action_name}] {action['query']}")
                print(f"     → {len(action['snippets'])} résultat(s)")
        
        if res['answer']:
            print(f"\n✅ RÉPONSE FINALE:")
            print(f"   {res['answer'][:200]}{'...' if len(res['answer']) > 200 else ''}\n")
        else:
            print(f"\n❌ PAS DE RÉPONSE GÉNÉRÉE\n")
        
        results_summary.append({
            'question': q,
            'answer': res['answer'],
            'num_actions': len(res['actions']),
            'stopped_by': res['stopped_by'],
        })

    # Résumé final
    print("\n" + "="*60)
    print("📋 RÉSUMÉ FINAL")
    print("="*60 + "\n")
    
    successful = sum(1 for r in results_summary if r['answer'] is not None)
    print(f"✅ Réponses générées: {successful}/{len(results_summary)}")
    print(f"❌ Échecs: {len(results_summary) - successful}/{len(results_summary)}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
