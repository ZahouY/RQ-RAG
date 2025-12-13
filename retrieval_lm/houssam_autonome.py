#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from duckduckgo_search import DDGS
    DUCK_AVAILABLE = True
except ImportError:
    DUCK_AVAILABLE = False
    DDGS = None


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


# -----------------------------
#  Agent autonome RQ-RAG
# -----------------------------
def rqrag_agent_autonome(
    model,
    tokenizer,
    question: str,
    max_steps: int = 4,
    max_new_tokens_step: int = 128,
    max_web_results: int = 3,
):
    """
    Implémente un RQ-RAG 'agent autonome' :
    - on ne force PAS les tokens d'action dans le prompt
    - le modèle décide quand émettre [S_...] ou [A_Response]
    - chaque [S_...] déclenche une recherche
    """
    history_text = build_agent_prompt(question)
    actions_log = []
    seen_actions = set()

    print("\n==============================")
    print("QUESTION :", question)
    print("==============================")

    for step in range(max_steps):
        print(f"\n===== STEP {step+1} =====")

        inputs = tokenizer(
            history_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens_step,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded_output = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
        )

        # 1) Chercher une réponse finale [A_Response]
        ans_matches = re.findall(
            r"\[A_Response\](.*?)\[EOS\]",
            decoded_output,
            flags=re.DOTALL,
        )
        if ans_matches:
            final_answer = ans_matches[-1].strip()
            print("🟢 [A_Response] détecté, arrêt.")
            print("\n--- FINAL ANSWER ---")
            print(final_answer)
            return {
                "answer": final_answer,
                "full_text": decoded_output,
                "actions": actions_log,
                "stopped_by": "A_Response",
            }

        # 2) Chercher la dernière action [S_...] produite
        act_matches = re.findall(
            r"\[(S_Rewritten_Query|S_Decomposed_Query|S_Disambiguated_Query)\](.*?)\[EOS\]",
            decoded_output,
            flags=re.DOTALL,
        )

        if act_matches:
            last_action, last_query = act_matches[-1]
            last_query = last_query.strip()

            if (last_action, last_query) in seen_actions:
                print("⚠️ Action déjà vue, on s'arrête (boucle potentielle).")
                break

            seen_actions.add((last_action, last_query))

            print(f"🟡 Nouvelle action: {last_action}")
            print(f"🔎 Query: {last_query}")

            # 3) Appel du moteur de recherche
            snippets = web_search(last_query, max_results=max_web_results)
            ev_text = format_evidences(snippets)

            actions_log.append({
                "action": last_action,
                "query": last_query,
                "snippets": snippets,
            })

            # 4) Réinjecter les évidences et continuer
            history_text = (
                decoded_output
                + "\n[R_Evidences]\n"
                + ev_text
                + "\n[/R_Evidences]\n"
            )
            continue

        # 3) Ni [A_Response], ni [S_...] détectés → on arrête
        print("⚠️ Aucun token [S_...] ni [A_Response] détecté, arrêt.")
        break

    print("\n🟥 Aucune [A_Response] produite.")
    return {
        "answer": None,
        "full_text": history_text,
        "actions": actions_log,
        "stopped_by": "max_steps_or_no_action",
    }


# -----------------------------
#  Chargement du modèle
# -----------------------------
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    model_name = os.environ.get("RQRAG_MODEL_NAME", "zorowin123/rq_rag_llama2_7B")


    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        "../models/rq_rag_llama2_7B",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "../models/rq_rag_llama2_7B",
        device_map="auto",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print("✅ Modèle chargé :", model_name)
    return model, tokenizer


# -----------------------------
#  Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test 'agent autonome' pour RQ-RAG (sans tree search)."
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
        help="Nombre max d'actions [S_...] avant d'arrêter.",
    )
    parser.add_argument(
        "--max_new_tokens_step",
        type=int,
        default=128,
        help="Nombre max de tokens générés par étape.",
    )
    parser.add_argument(
        "--max_web_results",
        type=int,
        default=3,
        help="Nombre max de résultats web par requête.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.question is None and args.questions_file is None:
        print("❌ Spécifie soit --question, soit --questions_file.")
        return

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

    for q in questions:
        res = rqrag_agent_autonome(
            model,
            tokenizer,
            q,
            max_steps=args.max_steps,
            max_new_tokens_step=args.max_new_tokens_step,
            max_web_results=args.max_web_results,
        )
        print("\n===== RÉSUMÉ =====")
        print("Stopped by:", res["stopped_by"])
        print("Answer:", res["answer"])
        print("Actions:")
        for a in res["actions"]:
            print("  -", a["action"], "→", a["query"])
        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
