from dataclasses import dataclass

from rag_engine.index.vector_store import RetrievedDocument

from llm_agent.agent import TennisAnalystAgent, _build_context_block
from llm_agent.live_features import LiveFeature, LiveFeatureSnapshot


def _snapshot():
    return LiveFeatureSnapshot(
        match_id="20240701-M-Wimbledon-R64-Cameron_Norrie-Jack_Draper",
        p1_name="Cameron Norrie", p2_name="Jack Draper",
        features=[
            LiveFeature("Score at point 40", "5-4, Norrie serving"),
            LiveFeature("ML-informed Markov engine estimate of Cameron Norrie win probability", "58.3%", is_estimate=True),
        ],
    )


def _docs():
    return [
        RetrievedDocument(
            doc_id="match:1", text="Norrie defeated Draper 7-6(3) 6-4 7-6(6) at Wimbledon 2024.",
            metadata={"doc_type": "match_summary"}, distance=0.1,
        ),
    ]


def test_build_context_block_tags_live_features_and_docs():
    text, tag_map = _build_context_block(_snapshot(), _docs())
    assert "[L1] Score at point 40: 5-4, Norrie serving" in text
    assert "[L2]" in text and "model estimate" in text
    assert "[D1] (match_summary) Norrie defeated Draper" in text
    assert tag_map["L1"] == "Score at point 40: 5-4, Norrie serving"
    assert tag_map["D1"].startswith("match_summary:")


def test_build_context_block_handles_no_sources():
    text, tag_map = _build_context_block(None, [])
    assert "no live features or retrieved context available" in text
    assert tag_map == {}


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeChat:
    def __init__(self, canned_answer):
        self._canned_answer = canned_answer
        self.sent_messages = []

    def send_message(self, message):
        self.sent_messages.append(message)
        return _FakeResponse(self._canned_answer)


class _FakeChats:
    def __init__(self, chat):
        self._chat = chat

    def create(self, **kwargs):
        return self._chat


class _FakeGenAIClient:
    def __init__(self, chat):
        self.chats = _FakeChats(chat)
        self.models = None


class _FakeGeminiClient:
    """Stands in for rag_engine.generate.GeminiClient without touching the real API."""

    def __init__(self, canned_answer):
        self.model = "gemini-3.5-flash"
        self._client = _FakeGenAIClient(_FakeChat(canned_answer))


class _FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, query, k=5, filters=None):
        return self._docs


def test_ask_reports_only_cited_sources():
    """The model's answer cites [L2] and [D1] but not [L1] -- sources_used should
    reflect exactly that, while sources_offered lists everything given to the model."""
    canned = "Norrie is favored, per the ML-informed model [L2], and won this exact 2024 match [D1]."
    agent = TennisAnalystAgent(
        vector_store=_FakeVectorStore(_docs()), gemini_client=_FakeGeminiClient(canned),
    )
    response = agent.ask("Who's likely to win?", live_features=_snapshot())

    assert response.answer == canned
    assert len(response.sources_used.live_features) == 1
    assert "ML-informed Markov" in response.sources_used.live_features[0]
    assert len(response.sources_used.retrieved_docs) == 1
    assert response.sources_used.retrieved_docs[0].startswith("match_summary:")
    assert set(response.sources_offered.keys()) == {"L1", "L2", "D1"}


def test_ask_reports_no_sources_used_when_model_cites_nothing():
    agent = TennisAnalystAgent(
        vector_store=_FakeVectorStore(_docs()),
        gemini_client=_FakeGeminiClient("Insufficient historical data for this."),
    )
    response = agent.ask("What happened in 1987?", live_features=_snapshot())
    assert response.sources_used.live_features == []
    assert response.sources_used.retrieved_docs == []


def test_ask_reuses_chat_across_turns_for_multiturn_state():
    fake_client = _FakeGeminiClient("some answer [D1]")
    agent = TennisAnalystAgent(vector_store=_FakeVectorStore(_docs()), gemini_client=fake_client)
    agent.ask("First question", live_features=_snapshot())
    agent.ask("Second question", live_features=_snapshot())
    chat = fake_client._client.chats._chat
    assert len(chat.sent_messages) == 2
