"""manual_eval.py — NOT a pytest file (no test_ functions, nothing asserted): a manual
review harness. Runs ~10 realistic tennis questions through the real agent (real
VectorStore retrieval against the built index, real Gemini calls) and prints each
answer plus its sources_used/sources_offered breakdown, so a human can read through and
flag hallucination or ungrounded claims before this is trusted anywhere
dashboard-facing or point_documents.py-adjacent.

Run directly: `python tests/manual_eval.py` (from llm_agent/, with GEMINI_API_KEY set
and rag_engine's index already built -- see rag_engine/README.md).

Deliberately mixes question types the grounding rules specifically target:
  - well-covered factual questions (should cite real docs/features confidently)
  - a question about a player/era NOT in the dataset (should say "insufficient
    historical data", not fabricate)
  - a live-probability question (should hedge as "the model estimates", name the
    engine, not state it as certain fact)
  - a multi-turn follow-up (tests that chat history carries over)
  - a comparison question spanning both live features and retrieved history
"""

from __future__ import annotations

from rag_engine.index.vector_store import VectorStore

from llm_agent.agent import TennisAnalystAgent
from llm_agent.live_features import LiveFeatureSnapshot, LiveFeature

QUESTIONS = [
    "What do you know about Cameron Norrie's record at Wimbledon?",
    "How has Jack Draper performed in five-set matches?",
    "Tell me about Bjorn Borg's 1977 Wimbledon campaign.",  # likely thin/absent coverage
    "Given the current live probabilities, who is more likely to win this match?",
    "What patterns show up in matches that go to a final-set tiebreak?",
    "How does Norrie's first-serve percentage compare across his charted matches?",
    "What's notable about break-point conversion in the retrieved matches?",
    "Following up on the previous answer -- has that trend held in more recent matches?",
    "Summarize the tactical matchup between the two players in the live features.",
    "What happened in the 1985 French Open final?",  # likely absent, tests fabrication resistance
]

LIVE_SNAPSHOT = LiveFeatureSnapshot(
    match_id="20240701-M-Wimbledon-R64-Cameron_Norrie-Jack_Draper",
    p1_name="Cameron Norrie", p2_name="Jack Draper",
    features=[
        LiveFeature("Score", "6-7(3), 4-6, Norrie 5-4* in the 3rd, serving"),
        LiveFeature(
            "Markov engine estimate of Cameron Norrie win probability", "44.2%", is_estimate=True,
        ),
        LiveFeature(
            "ML-informed Markov engine estimate of Cameron Norrie win probability",
            "47.8%", is_estimate=True,
        ),
    ],
)


def main() -> None:
    store = VectorStore()
    print(f"indexed docs: {store.count()}\n")
    agent = TennisAnalystAgent(vector_store=store)

    for i, question in enumerate(QUESTIONS, start=1):
        response = agent.ask(question, live_features=LIVE_SNAPSHOT)
        print(f"{'=' * 80}\nQ{i}: {question}\n{'-' * 80}")
        print(response.answer)
        print(f"\n[sources cited]  live={response.sources_used.live_features}")
        print(f"                 docs={response.sources_used.retrieved_docs}")
        print(f"[sources offered but not necessarily cited] {list(response.sources_offered.keys())}")
        print()


if __name__ == "__main__":
    main()
