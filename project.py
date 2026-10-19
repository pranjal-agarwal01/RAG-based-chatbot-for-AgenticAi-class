"""
NovaMart Customer Support Chatbot
=================================
A RAG (Retrieval-Augmented Generation) support chatbot with conversational memory.

1. Knowledge base: knowledge_base.md holds 18 customer-support questions and answers.
2. Retrieval:      TF-IDF search finds the FAQ entries closest to the customer's question.
3. Memory:         the conversation is kept for the session. Follow-up questions such as
                   "how long does it take?" are rewritten into standalone questions before
                   searching, and the recent chat is sent to the model with every question.
4. Grounding:      the model may answer only from the retrieved FAQ entries and says
                   clearly when the information is not available.
5. Interface:      a Streamlit chat app.

Run it with:  streamlit run project.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

KB_PATH = Path(__file__).parent / "knowledge_base.md"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 3          # how many FAQ entries are given to the model
MIN_SCORE = 0.10          # entries scoring below this are treated as not relevant
OFFLINE_MIN_SCORE = 0.15  # offline mode (no model) needs a clearer match before answering
FOLLOW_UP_SCORE = 0.30    # a follow-up matching an FAQ this well on its own is a new topic
MEMORY_TURNS = 5          # how many recent question/answer pairs the model sees

NOT_AVAILABLE = (
    "Sorry, I don't have that information in the NovaMart help guide. "
    "Please contact our support team at support@novamart.example "
    "or call 1800-123-4567 (9 AM to 9 PM)."
)
WELCOME = (
    "Hi! I'm the NovaMart support assistant. Ask me about orders, delivery, "
    "returns, refunds, payments, your account or warranty."
)
SAMPLE_QUESTIONS = [
    "How do I return an item?",
    "What are the delivery charges?",
    "Which payment methods do you accept?",
    "I forgot my password",
    "Do you sell groceries?",
]

SYSTEM_PROMPT = """You are the customer support assistant for NovaMart, an online electronics store in India.

Follow these rules:
1. Facts about NovaMart (policies, prices, time limits, steps, contact details) must come only from the FAQ entries below. Never use outside knowledge and never guess.
2. Use the earlier conversation to understand follow-up questions. You may repeat details the customer told you in this chat, such as their name, order ID or the product they bought.
3. If the answer is not in the FAQ entries or in this conversation, reply with exactly this text and nothing else:
{not_available}
4. If the customer only greets you or says thanks, reply politely in one short sentence.
5. Keep answers short and friendly (under 120 words). Use a numbered list for steps.
6. When you use an FAQ entry, end with its ID, for example: (Source: Q8)

FAQ entries:
{context}"""

REWRITE_PROMPT = """Rewrite the customer's latest message as one standalone question that makes sense without the conversation. Replace words like "it", "that" or "this" with what they refer to. If the message is already standalone, return it unchanged. Reply with the question only.

Conversation:
{conversation}

Latest message: {question}"""


# ---------------------------------------------------------------- knowledge base

def load_knowledge_base(path=KB_PATH):
    """Read the FAQ document. Every '### Q<n>.' heading starts a new entry."""
    entries = []
    category = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        heading = re.match(r"###\s+(Q\d+)\.\s+(.*)", line)
        if heading:
            entries.append({
                "id": heading.group(1),
                "category": category,
                "question": heading.group(2).strip(),
                "answer": "",
                "keywords": "",
            })
        elif line.startswith("## "):
            category = line[3:].strip()
        elif entries:
            if line.startswith("Keywords:"):
                entries[-1]["keywords"] = line[len("Keywords:"):].strip()
            else:
                entries[-1]["answer"] += line + "\n"
    for entry in entries:
        entry["answer"] = entry["answer"].strip()
    return entries


# ---------------------------------------------------------------- retrieval

STOP_WORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being am
do does did i me my we our you your he she it its they them their this that these those
there here what which who whom when where why how can could would should will shall may
might must have has had not no yes so than then too very just about into over please
hi hello hey tell know want get got any some
""".split())


def stem(word):
    """A tiny stemmer so that 'refunds', 'refunded' and 'refund' all match."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[:-len(suffix)]
            break
    if len(word) > 4 and word[-1] in "ey":
        word = word[:-1]
    return word


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if len(w) > 1 and w not in STOP_WORDS]


class Retriever:
    """TF-IDF search over the FAQ entries."""

    def __init__(self, entries):
        self.entries = entries
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False, sublinear_tf=True
        )
        texts = [f"{e['question']} {e['keywords']} {e['category']} {e['answer']}" for e in entries]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=TOP_K):
        """Return up to k (entry, score) pairs, best first, leaving out weak matches."""
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [(self.entries[i], float(scores[i])) for i in best if scores[i] >= MIN_SCORE]


# ---------------------------------------------------------------- memory

class ConversationMemory:
    """Remembers the current conversation.

    Every message is kept so the chat window can show it, but only the last
    MEMORY_TURNS question/answer pairs are sent to the model.
    """

    def __init__(self, max_turns=MEMORY_TURNS):
        self.max_turns = max_turns
        self.messages = []

    def add(self, role, content, **details):
        self.messages.append({"role": role, "content": content, **details})

    def recent(self):
        """The last few messages, in the format the chat model expects."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages[-2 * self.max_turns:]
        ]

    def last_search(self):
        """The search text used for the most recent support question, if any."""
        for message in reversed(self.messages):
            if message.get("search"):
                return message["search"]
        return None

    def clear(self):
        self.messages = []


FOLLOW_UP_WORDS = {"it", "its", "that", "this", "they", "them", "those", "these", "one", "ones"}


def looks_like_follow_up(question):
    """Offline rule of thumb: short questions, or ones using 'it'/'that'/'what about', depend on context."""
    text = question.lower().strip()
    words = re.findall(r"[a-z']+", text)
    return (
        len(words) <= 3
        or text.startswith(("what about", "how about", "and "))
        or any(w in FOLLOW_UP_WORDS for w in words)
    )


SMALL_TALK = {
    "hi": WELCOME,
    "hello": WELCOME,
    "hey": WELCOME,
    "good morning": WELCOME,
    "good afternoon": WELCOME,
    "good evening": WELCOME,
    "hi there": WELCOME,
    "hello there": WELCOME,
    "bye": "Thanks for contacting NovaMart. Have a great day!",
}
for thanks in ("thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ok", "okay", "ok thanks"):
    SMALL_TALK[thanks] = "You're welcome! Is there anything else I can help you with?"


def small_talk_reply(text):
    return SMALL_TALK.get(re.sub(r"[^a-z ]", "", text.lower()).strip())


# ---------------------------------------------------------------- chatbot

class SupportBot:
    def __init__(self, api_key=None, kb_path=KB_PATH):
        self.entries = load_knowledge_base(kb_path)
        self.retriever = Retriever(self.entries)
        self.client = None
        if api_key:
            from groq import Groq
            self.client = Groq(api_key=api_key)

    def reply(self, question, memory):
        """Answer one customer message and store the exchange in memory."""
        history = memory.recent()
        answer = small_talk_reply(question)
        search, sources = None, []
        if answer is None:
            search = self.search_text(question, memory)
            sources = self.retriever.search(search)
            answer = self.write_answer(question, sources, history)
        memory.add("user", question)
        memory.add("assistant", answer, sources=sources, search=search)
        return answer

    def search_text(self, question, memory):
        """Turn a follow-up such as 'how long does it take?' into a standalone question."""
        if not memory.messages:
            return question
        if self.client:
            try:
                rewritten = self.rewrite_with_llm(question, memory.recent())
                if rewritten:
                    return rewritten
            except Exception:
                pass  # fall back to the simple rule below
        previous = memory.last_search()
        if previous and looks_like_follow_up(question):
            alone = self.retriever.search(question, k=1)
            if not alone or alone[0][1] < FOLLOW_UP_SCORE:  # too vague to stand on its own
                return f"{previous} {question}"
        return question

    def rewrite_with_llm(self, question, history):
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        prompt = REWRITE_PROMPT.format(conversation=conversation, question=question)
        return self.chat([{"role": "user", "content": prompt}], max_tokens=600)

    def write_answer(self, question, sources, history):
        if self.client is None:
            return self.offline_answer(sources)
        context = "\n\n".join(
            f"[{entry['id']}] {entry['question']}\n{entry['answer']}" for entry, _ in sources
        ) or "(No FAQ entry matched this question.)"
        messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context, not_available=NOT_AVAILABLE),
        }]
        messages += history
        messages.append({"role": "user", "content": question})
        try:
            return self.chat(messages, max_tokens=1200) or self.offline_answer(sources)
        except Exception as error:  # bad key, no internet, rate limit...
            return (
                f"⚠️ The AI model could not be reached ({type(error).__name__}), "
                "so here is the closest answer from the help guide:\n\n"
                + self.offline_answer(sources)
            )

    def chat(self, messages, max_tokens):
        response = self.client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, max_completion_tokens=max_tokens
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def offline_answer(sources):
        """Without a model, reply with the best matching FAQ answer exactly as written.

        There is no model to judge whether a weak match really answers the question,
        so a stricter score is needed here than for the model's context.
        """
        if not sources or sources[0][1] < OFFLINE_MIN_SCORE:
            return NOT_AVAILABLE
        entry = sources[0][0]
        return f"**{entry['question']}**\n\n{entry['answer']}\n\n(Source: {entry['id']})"


# ---------------------------------------------------------------- Streamlit interface

def show_message(st, message, show_sources):
    st.markdown(message["content"])
    if show_sources and message.get("search"):
        sources = message["sources"]
        with st.expander(f"Retrieved FAQ entries ({len(sources)})"):
            st.caption(f"Searched for: {message['search']}")
            if not sources:
                st.write("No FAQ entry matched this question.")
            for entry, score in sources:
                st.markdown(
                    f"**{entry['id']}** · {entry['category']} · match {score:.2f}  \n{entry['question']}"
                )


def main():
    import streamlit as st

    st.set_page_config(page_title="NovaMart Support", page_icon="🛒")

    @st.cache_resource(show_spinner="Loading the help guide...")
    def load_bot(api_key):
        return SupportBot(api_key)

    bot = load_bot(os.getenv("GROQ_API_KEY", "").strip())
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    memory = st.session_state.memory

    with st.sidebar:
        st.header("🛒 NovaMart Support")
        if bot.client:
            st.success(f"Connected to Groq · {MODEL}")
        else:
            st.warning(
                "No GROQ_API_KEY found, so the bot runs in offline mode and replies "
                "with the best matching FAQ answer. Add a key to .env and restart the "
                "app for AI answers."
            )
        show_sources = st.toggle("Show retrieved FAQ entries", value=True)
        st.subheader("Try asking")
        for sample in SAMPLE_QUESTIONS:
            if st.button(sample, width="stretch"):
                st.session_state.pending = sample
        st.divider()
        st.caption(
            f"Knowledge base: {len(bot.entries)} FAQ entries. "
            f"Memory: the bot sees the last {MEMORY_TURNS} exchanges of this chat."
        )
        if st.button("Clear conversation", type="primary", width="stretch"):
            memory.clear()
            st.rerun()

    st.title("NovaMart Customer Support")
    st.caption(
        "Ask about orders, delivery, returns, refunds, payments, your account or warranty. "
        "Answers come only from the NovaMart help guide."
    )

    with st.chat_message("assistant"):
        st.markdown(WELCOME)
    for message in memory.messages:
        with st.chat_message(message["role"]):
            show_message(st, message, show_sources)

    question = st.chat_input("Type your question...")
    question = st.session_state.pop("pending", None) or question
    if question and question.strip():
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the help guide..."):
                bot.reply(question.strip(), memory)
            show_message(st, memory.messages[-1], show_sources)


if __name__ == "__main__":
    main()
