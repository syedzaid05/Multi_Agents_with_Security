# Multi-Agent AI Research System

A small, production-shaped research assistant built from cooperating agents and
a routed LLM gateway. Ask a question; the system plans searches, gathers cited
sources, analyses and critiques them, then writes a final, referenced answer —
all behind input/output guardrails, structured logging and latency tracking.

It ships with a **Streamlit UI** and a **headless CLI**, and it runs in
**demo mode with zero API keys** (a deterministic offline model answers), so you
can try it immediately.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Enabling real LLMs (free tiers first)](#enabling-real-llms-free-tiers-first)
- [Web search setup](#web-search-setup)
- [Running the app](#running-the-app)
- [Command-line usage](#command-line-usage)
- [Configuration reference](#configuration-reference)
- [Free vs paid services](#free-vs-paid-services)
- [Security: guardrails & prompt-injection defence](#security-guardrails--prompt-injection-defence)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Multi-agent pipeline** — Planner → Wikipedia → (optional) Web search →
  Analysis → Critic → Synthesis.
- **LLM gateway** with, per task:
  - **task-based routing** (`planner`, `research`, `analysis`, `critique`,
    `synthesis` each get their own model fallback chain),
  - **timeouts**, **retries** (exponential backoff + jitter),
  - **per-provider rate limiting** (async token bucket),
  - **model fallback** across providers, with an offline `echo` last resort.
- **Guardrails** — input validation/length limits, **prompt-injection
  detection**, untrusted-source fencing, and output checks (empty output,
  system-prompt leakage, citation coverage).
- **Source citations** — stable `[n]` numbering, de-duplication by URL, and a
  generated references section.
- **Structured logging** (JSON) and **per-stage latency tracking**.
- **Free-tier first** — Groq, Google Gemini, OpenRouter `:free` models, local
  Ollama, and keyless Wikipedia + DuckDuckGo search.
- **Streamlit UI** and **CLI**.
- **Pytest** suite (network-free, no keys required).

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   user query ─────▶│  Input guardrails (length, injection, ctrl)  │
                    └───────────────────────┬──────────────────────┘
                                            ▼
                                    ┌───────────────┐
                                    │ Planner agent │  (task: planner)
                                    └───────┬───────┘
                                            ▼
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
        ┌────────────────────┐                        ┌──────────────────────┐
        │  Wikipedia agent   │  (task: research)      │  Web search agent    │
        │  keyless REST API  │                        │  DuckDuckGo / Tavily │
        └─────────┬──────────┘                        └──────────┬───────────┘
                  └───────────────────────┬───────────────────────┘
                                          ▼   (fenced, untrusted content)
                                 ┌─────────────────┐
                                 │ Analysis agent  │  (task: analysis)
                                 └────────┬────────┘
                                          ▼
                                 ┌─────────────────┐
                                 │  Critic agent   │  (task: critique)
                                 └────────┬────────┘
                                          ▼
                                 ┌─────────────────┐
                                 │ Synthesis agent │  (task: synthesis)
                                 └────────┬────────┘
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │  Output guardrails + citations + references   │
                    └──────────────────────────────────────────────┘
```

Every agent call goes through the **LLM gateway**, which walks the task's
fallback chain (free/local models first, `echo` last):

```
groq:llama-3.3-70b-versatile → google:gemini-3.5-flash-lite → google:gemini-3.1-flash-lite
  → openrouter:meta-llama/llama-3.3-70b-instruct:free → ollama:llama3.2 → echo:echo-1
```

Key modules:

| Area | Module |
| --- | --- |
| LLM gateway / routing | `src/multi_agent_research/llm/` |
| Guardrails | `src/multi_agent_research/guardrails/` |
| Agents | `src/multi_agent_research/agents/` |
| Tools (Wikipedia, web) | `src/multi_agent_research/tools/` |
| Orchestration | `src/multi_agent_research/orchestration/pipeline.py` |
| Streamlit UI | `src/multi_agent_research/app.py` |
| CLI | `src/multi_agent_research/cli.py` |

---

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11 (uv can
install Python for you).

```powershell
# 1. Install dependencies (including the optional dev + web extras)
uv sync --all-extras

# 2. Create your env file
Copy-Item .env.example .env

# 3. Run the tests (no API keys needed)
uv run pytest

# 4. Launch the UI (works immediately in demo mode)
uv run streamlit run src/multi_agent_research/app.py
```

On macOS/Linux replace step 2 with `cp .env.example .env`.

> **Demo mode:** with no keys set, the gateway uses the offline `echo` model so
> the full pipeline (guardrails, citations, critic loop, latency tracking) is
> still exercised. Add a key below to get real answers.

---

## Enabling real LLMs (free tiers first)

Add **one or more** of the following to your `.env`. Routing prefers free
providers, so start with Groq (fastest to set up) and/or Google Gemini.

### Groq (free tier — recommended)

1. Create an account: <https://console.groq.com>
2. Open API keys: <https://console.groq.com/keys>
3. Create a key and add it to `.env`:

   ```dotenv
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

No paid plan required for the default models used here.

### Google Gemini (free tier)

1. Open Google AI Studio: <https://aistudio.google.com/app/apikey>
2. Create an API key.
3. Add to `.env`:

   ```dotenv
   GOOGLE_API_KEY=AIza...
   ```

### OpenRouter (`:free` models available)

1. Create an account: <https://openrouter.ai>
2. Create a key: <https://openrouter.ai/keys>
3. Add to `.env`:

   ```dotenv
   OPENROUTER_API_KEY=sk-or-...
   ```

The default OpenRouter entries use `:free` model variants. Using non-free
models there costs money.

### Ollama (local, free, no API key)

```powershell
# Install from https://ollama.com, then:
ollama serve
ollama pull llama3.2
```

`.env` defaults already point at `http://localhost:11434/v1`. You can also run
the app with **only** Ollama configured — no cloud keys needed.

### OpenAI / Together (optional, **paid**)

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
TOGETHER_API_KEY=...
```

These are **paid** services (Together offers a few free models; check current
pricing). They are only used if you put them in your routing chain.

After editing `.env`, just restart the app — no other changes needed.

---

## Web search setup

The web-search agent is **optional** and off the critical path.

- **DuckDuckGo (default, free, no key):** install the optional extra and set

  ```powershell
  uv sync --extra web
  ```

  ```dotenv
  WEB_SEARCH_BACKEND=duckduckgo
  ```

- **Tavily (free tier, then PAID):**

  1. Get a key at <https://tavily.com>
  2. Set:

     ```dotenv
     WEB_SEARCH_BACKEND=tavily
     TAVILY_API_KEY=tvly-...
     ```

- Disable entirely with `WEB_SEARCH_BACKEND=none` (or leave the UI checkbox off).

---

## Running the app

```powershell
uv run streamlit run src/multi_agent_research/app.py
```

The sidebar shows provider configuration (keys are masked), task routing, and a
web-search toggle. Results are split into tabs: **Plan, Sources, Analysis,
Critique, Guardrails, Metrics & Logs**.

---

## Command-line usage

```powershell
# Basic run (uses routing; falls back to echo if unconfigured)
uv run mar "What is the current state of fusion energy research?"

# With the web-search agent enabled
uv run mar "Latest developments in solid-state batteries" --web

# Print the redacted configuration and exit
uv run mar "ignored" --show-config
```

`mar` is the console script defined in `pyproject.toml`; the equivalent module
form is `uv run python -m multi_agent_research.cli "your query"`.

---

## Configuration reference

All configuration lives in environment variables / `.env` (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | – | Groq key (free tier) |
| `GOOGLE_API_KEY` | – | Google Gemini key (free tier) |
| `OPENROUTER_API_KEY` | – | OpenRouter key (has `:free` models) |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | – / api.openai.com | OpenAI (paid) |
| `TOGETHER_API_KEY` | – | Together (mostly paid) |
| `TAVILY_API_KEY` | – | Tavily web search (free tier then paid) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama endpoint |
| `WEB_SEARCH_BACKEND` | `duckduckgo` | `duckduckgo` \| `tavily` \| `none` |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `LOG_JSON` | `true` | JSON logs (`false` = human-readable) |
| `REQUEST_TIMEOUT_SECONDS` | `45` | Per-LLM-request timeout |
| `MAX_RETRIES` | `3` | Attempts per model before falling back |
| `REQUESTS_PER_MINUTE` | `30` | Per-provider rate limit |
| `MAX_INPUT_CHARS` | `4000` | Input guardrail limit |
| `MAX_OUTPUT_CHARS` | `12000` | Output guardrail limit |
| `LLM_ROUTES_JSON` | – | Override routing, e.g. `{"analysis": ["groq:llama-3.3-70b-versatile"]}` |
| `DEMO_MODE` | auto | Force demo mode on/off (auto = on when no keys) |

---

## Free vs paid services

| Service | Cost | Used for |
| --- | --- | --- |
| Wikipedia REST API | **Free**, no key | Primary research sources |
| DuckDuckGo search | **Free**, no key | Optional web search |
| Groq | **Free tier** | Primary LLM |
| Google Gemini | **Free tier** | Fallback LLM |
| OpenRouter `:free` models | **Free** (rate-limited) | Fallback LLM |
| Ollama (local) | **Free** | Fallback LLM |
| OpenAI | **PAID** | Optional |
| Together AI | Mostly **PAID** (some free models) | Optional |
| Tavily | **Free tier, then PAID** | Optional web search |

The system is fully functional using only free services; paid providers are
opt-in and never required.

---

## Security: guardrails & prompt-injection defence

- **Input guardrails** reject empty/oversized input, strip control characters,
  and **block high-severity prompt-injection** attempts (instruction override,
  system-prompt exfiltration, jailbreak phrasing, tool abuse, delimiter forging).
- **Untrusted-content fencing** — Wikipedia extracts and web snippets are
  wrapped in nonce-style delimiters (`<<<UNTRUSTED_SOURCE_START>>> …`) after
  stripping forged delimiters, and every agent's system prompt instructs the
  model to treat fenced text as **data only**.
- **Output guardrails** block empty output and system-prompt/secret leakage, and
  warn when an answer lacks `[n]` citations.
- **Secrets** live only in `.env`, which is **git-ignored**. Config snapshots
  shown in logs/UI are masked.

See `src/multi_agent_research/guardrails/` for the implementation and
`tests/test_guardrails.py` for the behaviours covered.

---

## Testing

```powershell
uv run pytest                 # run the suite
uv run pytest -q              # quiet
uv run pytest --cov           # with coverage (pytest-cov included)
```

Tests are hermetic: HTTP is mocked with `respx`, LLMs are replaced by fake/echo
providers, and no API keys or network access are required.

---

## Project layout

```
.
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── src/multi_agent_research/
│   ├── app.py                 # Streamlit UI
│   ├── cli.py                 # headless CLI
│   ├── config.py              # settings + provider specs
│   ├── factory.py             # composition root
│   ├── logging_config.py      # structured logging
│   ├── metrics.py             # spans / latency tracking
│   ├── citations.py           # source tracking
│   ├── state.py               # pipeline state
│   ├── llm/                   # gateway, routing, providers, rate limiting
│   ├── guardrails/            # input/output guards + injection defence
│   ├── tools/                 # Wikipedia + web search
│   ├── agents/                # planner/wiki/web/analysis/critic/synthesis
│   └── orchestration/         # pipeline
└── tests/                     # pytest suite
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Answers say `[DEMO]` | No LLM key configured. Add `GROQ_API_KEY` or `GOOGLE_API_KEY` to `.env` and restart. |
| "All providers failed" | Check keys/network; the error lists each model tried and why. |
| Web results missing | `uv sync --extra web`, set `WEB_SEARCH_BACKEND=duckduckgo`, enable the UI checkbox / `--web`. |
| Ollama timeouts | Ensure `ollama serve` is running and the model is pulled (`ollama pull llama3.2`). |
| `429` from a provider | Lower `REQUESTS_PER_MINUTE`, or rely on fallback; the gateway retries and moves on. |
| Want readable logs | Set `LOG_JSON=false`. |
