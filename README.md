# AgentSentinel

**Monitor. Evaluate. Diagnose. Improve.**

AgentSentinel is a provider-agnostic, framework-extensible observability and evaluation platform for AI applications. It is designed to capture normalized execution events, detect failures with application-owned rules, evaluate outcomes against explicit criteria, investigate incidents, benchmark performance, and present actionable insights in a developer-friendly dashboard.

## The problem

An AI application's final answer does not always explain what happened along the way. A workflow may take an unexpected path, call a tool incorrectly, return invalid output, or fail intermittently. Providers and agent frameworks expose different execution details, so useful analysis needs a normalized event format as well as consistent, application-owned definitions of failure and quality.

AgentSentinel uses **LangSmith as its primary tracing and debugging platform during initial development**. LangSmith is one observability integration, not the product itself: custom monitoring, failure detection, evaluation, and diagnostics consume normalized events and remain independent of LangSmith.

## Core features

### Initial MVP

- One LangGraph sample agent using Groq through LiteLLM.
- Calculator, knowledge-base, and mock-weather tools.
- LangSmith tracing for debugging.
- A provider/framework integration boundary that emits normalized execution events.
- Custom monitoring and failure detection independent of LangSmith.
- Evaluation, diagnostics, and a Streamlit dashboard.

### Future expansion

- Additional LLM providers and agent frameworks.
- Cross-model benchmarking and custom integrations.

> This repository is at the foundation stage. These are project goals, not claims that the features are already implemented.

## Technology stack

| Technology | Role |
|---|---|
| Python 3.11 | Application language and runtime |
| Conda | Environment management; reuse the existing `D:\AgentOps\venv` environment |
| LangGraph | Agent framework for the MVP; other frameworks can be added through adapters |
| Groq | Initial MVP model provider |
| LiteLLM | Initial unified gateway for standardized model access across supported providers |
| LangSmith | Primary tracing and debugging integration during initial development |
| Pydantic | Data validation and structured models |
| FastAPI | HTTP API |
| Streamlit | Interactive developer dashboard |
| pytest | Automated tests |

### Planned integration targets

| Category | MVP | Future targets |
|---|---|---|
| LLM providers | Groq through LiteLLM | OpenAI, Google Gemini, Anthropic Claude, Ollama, and other compatible providers |
| Agent frameworks | LangGraph | LangChain, CrewAI, AutoGen, and custom Python agents |

These are planned integrations, not implemented support. **Langfuse is not used.** Qdrant, Redis, and Kafka are outside the current scope.

## Architecture overview

```text
Streamlit dashboard (MVP) ----\
                               +--> AgentSentinel core
FastAPI (planned) -------------/     monitoring, detection,
                                     evaluation, diagnostics,
                                     benchmarking
                                              ^
                                              | normalized execution events
                                    Integration Layer
                              +---------------+----------------+
                              |                                |
                    Framework adapters                Provider gateway/adapters
                    LangGraph (MVP)                   LiteLLM (MVP)
                    LangChain, CrewAI,                Groq (MVP)
                    AutoGen, custom agents            OpenAI, Gemini, Claude,
                                                      Ollama, compatible providers
                              |                                |
                              +------- AI application --------+
                                              |
                                  LangSmith tracing/debugging
```

In simple terms: adapters translate framework- and provider-specific details into a shared event format. AgentSentinel's own code uses those events to determine failures and evaluate performance. LangSmith helps developers inspect traces; it does not define the product's monitoring or evaluation results. The integration layer is planned architecture and is not yet present in the scaffold.

## LangSmith tracing setup

When LangSmith tracing is enabled and a valid LangSmith API key is configured,
LangGraph automatically exports traces for agent runs; AgentSentinel does not
create a separate LangSmith client or callback. Set these values in your local
`.env` file to opt in:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=agentsentinel
```

`LANGSMITH_API_KEY` is your LangSmith project/workspace key, and
`LANGSMITH_PROJECT` selects the project where traces are organized. The variables
are already listed as empty placeholders in `.env.example`. Omit the tracing
settings or set `LANGSMITH_TRACING=false` to leave tracing disabled.

AgentSentinel passes its existing run ID as the `agentsentinel_run_id` LangChain
metadata value on the LangGraph invocation. LangGraph propagates this metadata
to child model and tool runs, so the trace can be correlated with persisted
AgentSentinel execution events that use the same run ID.

**Privacy:** LangSmith traces may capture prompts, tool inputs, and tool outputs.
AgentSentinel's custom-event redaction does not automatically apply to LangSmith
traces. Avoid sending sensitive data or enable tracing only when the captured
data is appropriate for your LangSmith workspace. Live trace-export verification
is optional; it requires tracing to be enabled and a valid `LANGSMITH_API_KEY`.

## Token usage and estimated costs

AgentSentinel records provider-reported input and output tokens per LLM call.
Estimated costs use USD rates per one million tokens and are stored with the
pricing version and rate provenance used for the calculation. Estimates do not
include taxes, discounts, caching, or other provider billing adjustments, and
are not invoices. Models without a verified exact model rate, or calls without
complete token usage, show cost as unavailable.

The initial catalog version is `2026-11-09-v1` (rates checked 09 Nov 2026):

| Exact model identifier | Input / output rate (USD per 1M tokens) | Source |
|---|---:|---|
| `anthropic/claude-sonnet-5` | $2 / $10 | [Anthropic API pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| `gemini/gemini-4-argon` | $2 / $10 through 30 Nov 2026 | [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) |

Pricing is intentionally exact-match and versioned. Groq, OpenAI, other
Anthropic/Gemini models, and any unlisted model identifiers remain unpriced
until their model-specific rates are verified and added to the catalog.

## Project structure

```text
AgentSentinel/
├── app/
│   ├── agent/          # Agent workflow orchestration
│   ├── api/            # FastAPI endpoints
│   ├── dashboard/      # Streamlit interface
│   ├── detection/      # Failure detection rules
│   ├── diagnostics/    # Incident explanations and next steps
│   ├── evaluation/     # Evaluation criteria and results
│   ├── integrations/   # Planned provider/framework adapters and event schemas
│   └── monitoring/     # Application-owned monitoring
├── core/               # Shared configuration and foundational types
├── data/               # Local or sample data
├── docs/               # Project documentation
├── knowledge_base/     # Future curated reference material, if needed
├── tests/              # pytest tests
├── .env.example        # Names of required local settings; no real secrets
├── PROJECT_CONTEXT.md  # Detailed architecture and contributor context
└── README.md
```

`app/integrations/` describes the intended future layer; it has not been created in the current foundation scaffold.

## Getting started (initial draft)

The project foundation is not yet a working application. These steps document the intended local workflow; dependency installation and the project dependency list have not been validated as a complete setup.

1. Clone the repository and open its directory.
2. Reuse the existing Conda environment. Do **not** create another environment:

   ```powershell
   conda activate D:\AgentOps\venv
   python --version
   ```

   The expected Python version is 3.11.

3. After the project dependencies and installation instructions have been reviewed, install them using the documented project workflow:

   ```powershell
   pip install -r requirements.txt
   ```

4. Copy the variable names from `.env.example` into a local `.env` file and fill in credentials locally. Never commit `.env` or paste API keys into source code, logs, or traces.
5. Once tests exist, run them with:

   ```powershell
   pytest
   ```

Do not install packages or add real credentials as part of this initial scaffold.

## Planned development roadmap

### Day 1 — Integrated sample agent

- Define the minimum normalized event and run data needed for the MVP.
- Build one LangGraph agent with calculator, knowledge-base, and mock-weather tools.
- Route Groq model calls through LiteLLM and connect LangSmith tracing.
- Add focused tests for event normalization and core workflow behavior.

### Day 2 — Independent insights and dashboard

- Add a small set of custom monitoring signals and failure categories over normalized events.
- Implement focused evaluation and evidence-based diagnostics with tests.
- Build a minimal Streamlit dashboard for runs and findings.
- Review privacy and verify that custom analysis works independently of LangSmith.

To keep two days realistic, the MVP is limited to one framework, one provider, three simple tools, and a small set of checks. Additional providers and frameworks, cross-model benchmarking, custom integrations, production-scale persistence, and a full API are future expansion. See [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) for detailed responsibilities, security practices, and guidance for future coding assistants.

## Development principles

- Keep modules small and behavior testable.
- Make failures explicit and distinguish evidence from AI-generated interpretation.
- Protect credentials and minimize sensitive data sent to external services.
- Add infrastructure only when there is a concrete need.
- Treat all roadmap features as planned until implemented and tested.
