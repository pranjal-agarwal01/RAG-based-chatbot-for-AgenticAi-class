# NovaMart Customer Support Chatbot (RAG + Conversational Memory)

A support chatbot for **NovaMart**, a fictional online electronics store. Customers ask about orders, delivery, returns, refunds, payments, their account or warranty. The bot finds the matching entries in a small FAQ and answers only from them. It also remembers the conversation, so follow-up questions work.

## Files

| File                  | What it does                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `knowledge_base.md` | The knowledge base: 18 customer-support questions and answers in 5 categories                                      |
| `project.py`        | The chatbot: FAQ loading, TF-IDF retrieval, conversation memory, grounded answers and the Streamlit chat interface |
| `test_project.py`   | 38 automated checks (no API key needed)                                                                            |
| `requirements.txt`  | Python packages                                                                                                    |
| `.env.example`      | Template for your Groq API key                                                                                     |

## How it works

```
question ──► memory: is this a follow-up? ──► rewrite into a standalone question
                                                   │
                                                   ▼
             FAQ (knowledge_base.md) ──► TF-IDF search ──► top 3 matching entries
                                                   │
                                                   ▼
             system prompt (rules + entries) + recent chat + question ──► Groq LLM ──► answer + source IDs
```

1. **Knowledge base**: `knowledge_base.md` is split into one chunk per question (`### Q1.` … `### Q18.`). Each entry has a `Keywords:` line with other words customers use, which helps the search.
2. **Retrieval (RAG)**: every entry is turned into a TF-IDF vector (scikit-learn). The question is compared to all entries with cosine similarity, and the top 3 entries scoring at least 0.10 are passed to the model.
3. **Conversational memory**: the whole chat is kept in the Streamlit session. Before searching, a follow-up such as *"How long does it take?"* is rewritten by the model into a standalone question (*"How long does a refund take?"*). The last 5 question/answer pairs are also sent with every request, so the bot can use details you gave earlier, such as your order ID.
4. **Grounded answers**: the system prompt only allows facts from the retrieved entries. If they don't contain the answer, the bot replies with a fixed "not available" message plus the support contact details. Every answer ends with its source ID, for example `(Source: Q8)`.
5. **Interface**: a Streamlit chat with sample questions, a panel showing the retrieved FAQ entries and their match scores, and a button to clear the conversation.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate it:

- Windows Command Prompt: `.venv\Scripts\activate.bat`
- Windows PowerShell: `.venv\Scripts\Activate.ps1` (if it says running scripts is disabled, first run `Set-ExecutionPolicy -Scope Process Bypass`, which only affects that window)
- Mac/Linux: `source .venv/bin/activate`

Then install the packages:

```bash
pip install -r requirements.txt
```

Get a free API key from https://console.groq.com/keys, copy `.env.example` to `.env` and paste the key:

```
GROQ_API_KEY=gsk_your_key_here
```

If the app is already running, stop it (Ctrl+C) and start it again after editing `.env`. Never share your `.env` file: it contains your key.

## Run

```bash
streamlit run project.py
```

The app opens at http://localhost:8501.

**Without an API key** the app still runs, in offline mode: it shows the best-matching FAQ answer word for word, or the "not available" message when no entry matches clearly. Follow-ups use a simple rule there: a short or vague question (for example one with "it" or "that") is combined with the previous question, unless it clearly matches an FAQ entry on its own. Offline mode cannot judge whether the closest FAQ really answers the question, and it cannot recall details you typed earlier, so use a key to see the full behaviour.

## Try this conversation (with a Groq API key)

| You type                                   | What the bot should do                                               |
| ------------------------------------------ | -------------------------------------------------------------------- |
| Hi, my order NM48151623 hasn't arrived yet | Explain the late-delivery steps (Q6)                                 |
| What if it shows delivered?                | Follow-up: answer from Q6 about packages marked delivered            |
| How do I return an item?                   | Return steps (Q8)                                                    |
| How long will the refund take?             | Refund timelines (Q9)                                                |
| What was my order number?                  | Remembers NM48151623 from the first message                          |
| Do you sell groceries?                     | Says the information is not available and gives the support contacts |

In offline mode rows 1 to 4 and 6 behave the same, but row 5 gets the "not available" message because only the model can read the earlier chat.

## Tests

```bash
python -m pytest -q
```

The tests check the knowledge base, retrieval for every FAQ, unrelated questions, memory, follow-up rewriting, grounding and the fallback when the model can't be reached. The model is replaced by a fake, so no key or internet connection is needed.

## Changing the support role

Everything specific to NovaMart is in `knowledge_base.md` plus the texts in `project.py` that mention NovaMart: `NOT_AVAILABLE`, `WELCOME`, `SAMPLE_QUESTIONS`, `SYSTEM_PROMPT`, the "bye" reply in `SMALL_TALK` and the titles in `main()` (search the file for "NovaMart"). Replace those with, say, university helpdesk FAQs and the same code works for that role.

## Limitations

- TF-IDF matches words, not meaning. A question worded very differently from the FAQ (for example *"I'd like a blue one instead of black"*) may not find the right entry. The `Keywords:` lines and the follow-up rewriting help with this.
- Memory lasts only for the current browser session. Clearing the chat or refreshing the page starts a new conversation.
