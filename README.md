# Autonomous-Task-Planning-System

[![CI](https://github.com/Aditya-k63/Autonomous-Task-Planning-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditya-k63/Autonomous-Task-Planning-System/actions/workflows/ci.yml)

An AI agent that takes a high-level goal, breaks it into smaller tasks, figures out the order to do them in, and executes them one by one — using real tools like file operations, shell commands, and web search.

But here's the interesting part: if the agent gets stuck or needs your input mid-execution, it pauses and shows you **3 clear options** to choose from in a clean UI. You pick one, and it keeps going.

**And the part that matters more than the agent: this repo is an evaluation harness for it.** 15 benchmark tasks with programmatic verifiers, 11 fault classes injected at the tool and model seams, a permission model that gates every call, and a CI gate that fails the build when task success drops. You cannot claim the agent works without a number that says so.

## How It Works

You give it a goal like *"Build a REST API with authentication and tests"*. Here's what happens behind the scenes:

1. **Planning** — An LLM breaks your goal into concrete subtasks with dependencies. It figures out that you need to create the database schema before you can add authentication, and so on.

2. **Execution** — The agent picks up each task and uses tools to complete it. It can read and write files, run shell commands, search the web, and fetch URLs.

3. **Clarification** — When the agent hits something ambiguous (like "which database should I use?"), it doesn't guess. Instead, it shows you a card with 3 options:

   ```
   ┌─────────────────────────────────────────────┐
   │  CLARIFICATION NEEDED                       │
   │                                             │
   │  Which database should I use for auth?      │
   │                                             │
   │  ┌─────────────────────────────────────┐    │
   │  │ a │ SQLite + bcrypt                 │    │
   │  │   │ Simple, file-based              │    │
   │  ├───┼─────────────────────────────────┤    │
   │  │ b │ PostgreSQL + JWT                │    │
   │  │   │ Production-ready, scalable      │    │
   │  ├───┼─────────────────────────────────┤    │
   │  │ c │ Firebase Auth                   │    │
   │  │   │ Fully managed, no backend       │    │
   │  └─────────────────────────────────────┘    │
   └─────────────────────────────────────────────┘
   ```

   You click an option (or type your own answer), and the agent continues with that context.

4. **Review** — After each task, an LLM reviews whether it was actually completed correctly. If not, it retries or replans.

5. **Completion** — Once all tasks are done, you get a summary of everything that was accomplished.

## The Tech

- **LangGraph** — Orchestrates the agent loop using a state machine (StateGraph). Each node (planner, executor, reviewer, router) is a step in the graph, with conditional edges that determine what happens next.

- **FastAPI** — Powers the API endpoints and serves the web UI. Everything runs as a single service.

- **LangChain + Groq** — The LLM backbone. Uses Groq's free tier for fast inference (currently llama-3.3-70b via OpenRouter). Supports OpenRouter free models too.

- **Model fallback chain** — Endpoints are tried in order: OpenRouter free models first, then Groq, then xAI Grok. A transient failure retries on the same endpoint before the chain moves on, so a rate-limited model never sinks a run.

- **Custom Tool System** — Tools are registered in a plugin-style registry. Each tool is just a function that takes a string and returns a string. New tools can be added by dropping a file in the tools folder.

## Reliability, Safety & Evaluation

Most agent demos show the happy path. This repo is built around the failure path.

### Permission model

Every tool call is classified before it runs. Shell commands are scored
**allow / ask / deny** from read-only and write allowlists, hard-deny
commands, and dangerous-pattern regexes:

| Decision | Meaning | Example |
|---|---|---|
| `allow` | runs immediately | `ls -la`, `cat out/report.md`, `grep -r TODO .` |
| `ask` | needs confirmation | `rm -rf build`, `git commit`, `curl https://…` |
| `deny` | refused, with a reason | `rm -rf /`, `curl … \| sh`, `shutdown now`, `sudo …` |

A refusal returns an explanatory error the agent can act on, and every
decision is recorded as a permission event. The policy runs in three
modes: `off`, `ask` (default), and `enforce`.

### Fault tolerance

| Layer | Behaviour |
|---|---|
| Schema validation | Every tool declares a JSON Schema; invalid arguments are rejected before dispatch |
| Retry budget | Transient tool failures retry with exponential backoff, up to a per-tool budget |
| Circuit breaker | A tool that keeps failing is opened and skipped for a cooldown instead of burning the step budget |
| Output isolation | Tool output is wrapped in `<<<TOOL_OUTPUT_UNTRUSTED>>>` markers so fetched content cannot be read as instructions |
| Native tool calling | Tools are bound to the model as functions, so the model emits structured calls rather than hand-written JSON |
| Context budget | The conversation is trimmed to the system prompt plus the most recent tool exchanges, so a long task does not pay for its own history |
| Model fallback | The chain walks OpenRouter → Groq → xAI, retrying transient errors on each endpoint before moving on |
| Rate-limit awareness | A provider that says "try again in 18m" is skipped rather than slept through, and quarantined so later calls skip it without a round trip |
| Goal-aware stopping | The reviewer reports whether the original goal is already met, and the agent stops instead of grinding through the rest of the plan |

The last four exist because they were needed to make the agent work
against a free-tier provider in practice, not because they are on a
checklist. The token accounting below is what made the cost visible.

### What the numbers cost

Two changes came straight out of measuring rather than guessing: the
planner was producing **nineteen tasks for a one-file goal**, and the
executor was resending its whole history on every call. Fixing the first
(short plans, plus stopping once the goal is satisfied) and the second
(a context budget) took the first benchmark task from **eight tool
calls and sixty-four seconds** down to **two tool calls and fourteen
seconds**, and cut input tokens from roughly thirty thousand to **6,540**.

On a free tier with a 200,000 token daily allowance, that is the
difference between a benchmark that runs and one that does not.

### Evaluation harness

```bash
# Reference-solver run — no API key needed, green in CI
python -m eval.cli --baseline --oracle
python -m eval.cli --report --oracle

# Live model run
python -m eval.cli --baseline
python -m eval.cli --all-faults
python -m eval.cli --report

# Gate the build
python -m eval.gate --min-success-rate 1.0
```

**15 benchmark tasks, every one with a programmatic verifier** — no LLM
judge is used for ground truth, so a run is deterministic and cheap enough
for CI. The tasks cover file creation, transformation, editing, search,
data aggregation, shell use, and multi-step chains.

Each task is checked twice in the test suite: it must **fail** on an empty
workspace (otherwise it is free) and **pass** when the work is done
(otherwise the verifier measures nothing).

**11 fault classes** are injected at two seams:

| Scope | Faults |
|---|---|
| Tool | `tool_timeout`, `tool_exception`, `tool_malformed_output`, `tool_empty_result`, `tool_slow`, `tool_permission_denied`, `tool_prompt_injection` |
| Model | `llm_malformed_json`, `llm_refusal`, `llm_truncated`, `llm_wrong_tool` |

`tool_prompt_injection` is the one that matters most: the agent has
`web_search` and `fetch_url`, so untrusted content really can try to
redirect it. The fault injects a hostile instruction into tool output and
the run records whether the agent obeyed it.

## Measured results

| Fault | Tasks | Success | Success rate | Mean steps |
|---|---:|---:|---:|---:|
| tool_timeout | 15 | 0 | 0.0% | 3.00 |
| tool_exception | 15 | 0 | 0.0% | 3.00 |
| tool_malformed_output | 15 | 0 | 0.0% | 3.00 |
| tool_empty_result | 15 | 0 | 0.0% | 3.00 |
| tool_slow | 15 | 1 | 6.7% | 3.00 |
| tool_permission_denied | 15 | 0 | 0.0% | 3.00 |
| tool_prompt_injection | 15 | 0 | 0.0% | 3.00 |
| llm_malformed_json | 15 | 0 | 0.0% | 2.00 |
| llm_refusal | 15 | 0 | 0.0% | 2.00 |
| llm_truncated | 15 | 0 | 0.0% | 2.00 |
| llm_wrong_tool | 15 | 0 | 0.0% | 7.00 |

The full report, including per-task-kind breakdowns and failure detail, is regenerated by `python -m eval.cli --report` and committed to `benchmarks/results/report.md`.

### The harness found four real bugs

Building the harness surfaced defects in the agent that a demo never
would have:

1. **Infinite loop on malformed output.** A `TOOL_CALL` that failed to
   parse did not count against the step budget, so a model emitting bad
   JSON looped forever. The same applied to `CLARIFY` parsing. Both are
   fixed, and the runner now enforces a per-task timeout.
2. **Dispatch layer assumed strings.** A tool returning a dict broke the
   failure check. Tool output is now coerced and the check is type-safe.
3. **The model was speaking a protocol the agent did not understand.**
   It emitted native function calls, but the executor only parsed a
   `TOOL_CALL: {json}` text format, so every step was wasted and the
   run never finished. The agent now binds tools as functions and
   handles both the native and the legacy text protocol.
4. **The agent could not run on Windows at all.** `app/tools/shell.py`
   imported `resource`, which is POSIX-only, so the module failed to
   import and six of the repo's own tests failed. The shell tool is now
   cross-platform, with the POSIX resource limits applied only where
   they exist.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Aditya-k63/Autonomous-Task-Planning-System.git
cd Autonomous-Task-Planning-System

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY (free models work),
# and optionally GROQ_API_KEY or XAI_API_KEY as fallbacks

# Run it
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Running the evaluation

```bash
# No API key required — uses the scripted reference solver
python -m eval.cli --baseline --oracle

# With a live model
python -m eval.cli --baseline
python -m eval.cli --all-faults
python -m eval.cli --report
```

### Using Docker

```bash
docker build -t task-planner-agent .
docker run -p 8000:8000 --env-file .env task-planner-agent
```

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | The web UI |
| `/start` | POST | Send a goal, get back a session with tasks |
| `/status/{id}` | GET | Check what the agent is doing right now |
| `/clarify/{id}` | GET | Get the pending question (if any) |
| `/clarify` | POST | Answer a clarification question |
| `/sessions` | GET | List all active sessions |
| `/health` | GET | Health check |

## Project Structure

```
Autonomous-Task-Planning-System/
├── app/
│   ├── main.py              # FastAPI app and endpoints
│   ├── config.py            # Environment settings
│   ├── state.py             # Data schemas (AgentState, Task, PermissionEvent)
│   ├── graph.py             # LangGraph wiring — connects all nodes
│   ├── llm.py               # LLM provider setup with the fallback chain
│   ├── llm_models.py        # Provider → model catalogue
│   ├── workspace.py         # Sandbox root resolution and path containment
│   ├── context.py           # Context budget — keeps the prompt bounded
│   ├── nodes/
│   │   ├── planner.py       # Turns goals into task lists
│   │   ├── executor.py      # Runs tools behind the permission gate
│   │   ├── reviewer.py      # Checks if tasks were done correctly
│   │   ├── router.py        # Decides what to do next
│   │   └── clarifier.py     # Handles the 3-option question flow
│   ├── permissions/
│   │   ├── bash_security.py # Command risk classifier (allow/ask/deny)
│   │   ├── model.py         # Permission policy and request types
│   │   └── explainer.py     # Turns a request into a user-facing explanation
│   ├── tools/
│   │   ├── registry.py      # Validation, retries, circuit breaker, dispatch
│   │   ├── file_ops.py      # read, write, edit, list, exist, search, glob, grep
│   │   ├── shell.py         # Run shell commands safely
│   │   ├── search.py        # Search the web
│   │   ├── web.py           # Fetch URLs
│   │   └── output.py        # Untrusted-output isolation markers
│   └── memory/
│       └── sessions.py      # Store session data
├── eval/                    # The evaluation harness
│   ├── tasks.py             # 15 benchmark tasks + programmatic verifiers
│   ├── faults.py            # 11 fault classes
│   ├── runner.py            # Runs tasks, installs faults, records metrics
│   ├── metrics.py           # Aggregation and before/after comparison
│   ├── report.py            # Markdown report rendering
│   ├── oracle.py            # Scripted reference solver (CI without keys)
│   ├── gate.py              # CI success-rate gate
│   └── cli.py               # Command-line entry point
├── benchmarks/results/      # Generated run artefacts
├── static/
│   └── index.html           # The web UI
├── tests/                   # Unit and end-to-end tests
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## License

MIT
