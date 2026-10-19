"""Checks for the support chatbot. Run with:  python -m pytest -q

No API key is needed: the model is replaced by a small fake.
"""

from types import SimpleNamespace

import pytest

import project
from project import NOT_AVAILABLE, ConversationMemory, SupportBot


class FakeModel:
    """Stands in for the Groq client: returns canned replies and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


@pytest.fixture(scope="module")
def offline_bot():
    return SupportBot()


def bot_with_model(replies):
    bot = SupportBot()
    bot.client = FakeModel(replies)
    return bot


# ---------------------------------------------------------------- knowledge base

def test_knowledge_base_has_10_to_20_entries(offline_bot):
    entries = offline_bot.entries
    assert 10 <= len(entries) <= 20
    assert [e["id"] for e in entries] == [f"Q{i}" for i in range(1, len(entries) + 1)]
    for entry in entries:
        assert entry["question"] and entry["answer"] and entry["category"]
        assert "Keywords:" not in entry["answer"]


# ---------------------------------------------------------------- retrieval

@pytest.mark.parametrize("question, expected", [
    ("Where is my order?", "Q1"),
    ("I want to cancel my order", "Q2"),
    ("I entered the wrong address", "Q3"),
    ("How much is delivery?", "Q4"),
    ("Do you ship to the USA?", "Q5"),
    ("My package hasn't arrived yet", "Q6"),
    ("Can I return opened earphones?", "Q7"),
    ("How do I send an item back?", "Q8"),
    ("When will I get my money back?", "Q9"),
    ("The product arrived broken", "Q10"),
    ("My headphones are not working", "Q10"),
    ("Can I pay by credit card?", "Q12"),
    ("Can I exchange it for a different colour?", "Q11"),
    ("Do you accept cash on delivery?", "Q12"),
    ("Money got debited but the order failed", "Q13"),
    ("Can I pay in instalments?", "Q14"),
    ("My coupon is not working", "Q15"),
    ("I forgot my password", "Q16"),
    ("Is there a warranty?", "Q17"),
    ("What is your phone number?", "Q18"),
])
def test_retrieval_finds_the_right_faq(offline_bot, question, expected):
    results = offline_bot.retriever.search(question)
    assert results, f"nothing retrieved for {question!r}"
    assert results[0][0]["id"] == expected


@pytest.mark.parametrize("question", [
    "Who won the cricket world cup?",
    "What is the capital of France?",
    "Write me a poem",
])
def test_unrelated_questions_retrieve_nothing(offline_bot, question):
    assert offline_bot.retriever.search(question) == []


# ---------------------------------------------------------------- memory

def test_memory_keeps_everything_but_sends_only_recent_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(4):
        memory.add("user", f"question {i}")
        memory.add("assistant", f"answer {i}", search=f"question {i}")
    assert len(memory.messages) == 8
    assert [m["content"] for m in memory.recent()] == ["question 2", "answer 2", "question 3", "answer 3"]
    assert memory.last_search() == "question 3"
    memory.clear()
    assert memory.messages == [] and memory.last_search() is None


def test_offline_follow_up_uses_the_previous_question(offline_bot):
    memory = ConversationMemory()
    offline_bot.reply("How do I return an item?", memory)
    offline_bot.reply("How long does it take?", memory)
    assert memory.messages[-1]["search"] == "How do I return an item? How long does it take?"
    offline_bot.reply("How do I reset my password?", memory)
    assert memory.messages[-1]["search"] == "How do I reset my password?"


def test_offline_short_question_on_a_new_topic_is_not_merged(offline_bot):
    memory = ConversationMemory()
    offline_bot.reply("Do you offer EMI?", memory)
    offline_bot.reply("What about cash on delivery?", memory)
    assert memory.messages[-1]["sources"][0][0]["id"] == "Q12"


def test_follow_up_is_rewritten_by_the_model_before_searching():
    bot = bot_with_model([
        "Go to My Orders and click Return. (Source: Q8)",
        "How long does a refund take after I return an item?",
        "Refunds start after a quality check... (Source: Q9)",
    ])
    memory = ConversationMemory()
    bot.reply("How do I return an item?", memory)
    bot.reply("How long will it take to get my money?", memory)

    first, rewrite, second = bot.client.requests
    assert len(first["messages"]) == 2              # system prompt + question, no history yet
    assert "How do I return an item?" in rewrite["messages"][0]["content"]
    assert memory.messages[-1]["search"] == "How long does a refund take after I return an item?"
    assert [m["content"] for m in second["messages"][1:3]] == [
        "How do I return an item?",
        "Go to My Orders and click Return. (Source: Q8)",
    ]
    assert "[Q9]" in second["messages"][0]["content"]


def test_empty_rewrite_falls_back_to_the_follow_up_rule():
    bot = bot_with_model(["Go to My Orders and click Return. (Source: Q8)", "", "Refunds start after... (Source: Q9)"])
    memory = ConversationMemory()
    bot.reply("How do I return an item?", memory)
    bot.reply("How long does it take?", memory)
    assert memory.messages[-1]["search"] == "How do I return an item? How long does it take?"


def test_model_can_use_details_from_earlier_in_the_chat():
    bot = bot_with_model(["Sorry to hear that! ...", "What is my order ID?", "Your order ID is NM10293847."])
    memory = ConversationMemory()
    bot.reply("My order NM10293847 is late", memory)
    bot.reply("What was my order ID again?", memory)
    last_request = bot.client.requests[-1]["messages"]
    assert any("NM10293847" in m["content"] for m in last_request if m["role"] == "user")


# ---------------------------------------------------------------- grounding

def test_offline_answer_comes_from_the_faq(offline_bot):
    memory = ConversationMemory()
    answer = offline_bot.reply("Which payment methods do you accept?", memory)
    faq = next(e for e in offline_bot.entries if e["id"] == "Q12")
    assert faq["answer"] in answer and "(Source: Q12)" in answer


@pytest.mark.parametrize("question", [
    "What is the capital of France?",
    "Do you sell groceries?",
    "What was my order number?",
])
def test_offline_bot_says_when_information_is_missing(offline_bot, question):
    assert offline_bot.reply(question, ConversationMemory()) == NOT_AVAILABLE


def test_prompt_contains_only_retrieved_entries_and_the_refusal_rule():
    bot = bot_with_model(["We accept UPI, cards... (Source: Q12)"])
    bot.reply("Which payment methods do you accept?", ConversationMemory())
    system = bot.client.requests[0]["messages"][0]["content"]
    assert "[Q12]" in system and "[Q7]" not in system
    assert NOT_AVAILABLE in system


def test_model_is_told_when_nothing_matched():
    bot = bot_with_model([NOT_AVAILABLE])
    answer = bot.reply("Who won the cricket world cup?", ConversationMemory())
    assert "(No FAQ entry matched this question.)" in bot.client.requests[0]["messages"][0]["content"]
    assert answer == NOT_AVAILABLE


def test_model_error_falls_back_to_the_faq_answer():
    bot = bot_with_model([ConnectionError("no internet")])
    answer = bot.reply("Which payment methods do you accept?", ConversationMemory())
    assert answer.startswith("⚠️") and "(Source: Q12)" in answer


def test_small_talk_does_not_search(offline_bot):
    memory = ConversationMemory()
    assert offline_bot.reply("Hi!", memory) == project.WELCOME
    assert memory.messages[-1]["search"] is None
    offline_bot.reply("Thanks a lot!", memory)
    assert memory.messages[-1]["search"] is None
