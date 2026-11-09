# AgentSentinel Project Context

## Project overview

AgentSentinel is a provider-agnostic, framework-extensible platform for observing, evaluating, diagnosing, and benchmarking AI applications. Its goal is to make application behavior easier to understand: what happened during a run, whether it succeeded, how well it performed, and what an engineer can do about a failure.

The project is being built as a learning exercise with maintainable boundaries between integrations, normalized execution data, custom analysis, and user-facing interfaces. The MVP will use one agent framework and one model provider to prove those boundaries; future integrations should not require rewriting the monitoring and evaluation core.

## Problem statement

AI applications can fail in ways that are difficult to see from a final answer alone. A run may call the wrong tool, take an unexpected path, return an invalid result, exceed a latency or cost target, or fail intermittently. Each provider and agent framework may expose different trace formats and concepts. AgentSentinel needs a common representation of execution events so its product features can work across those differences.

AgentSentinel aims to turn normalized execution events into useful operational signals, evaluations, benchmarks, and human-readable explanations. LangSmith is the primary tracing and debugging platform during initial development, but it is one observability integration—not the product itself. AgentSentinel's own monitoring, failure detection, evaluation, and diagnostics must remain independent of LangSmith.

## Main objectives

1. Consume normalized execution events instead of binding product features to a specific provider or framework.
2. Provide an integration layer for LLM providers, agent frameworks, and trace-event normalization.
3. Detect and categorize failures using application-owned rules.
4. Evaluate executions using explicit, testable criteria.
5. Produce concise diagnostics that help a developer decide what to investigate next.
6. Benchmark agent or model performance using comparable execution data as the platform expands.
7. Present runs and actionable insights through a dashboard and, as the API is developed, service endpoints.
8. Use LangSmith as the primary tracing and debugging platform during initial development without making custom product logic depend on it.

## High-level architecture

```text
User / developer
      +--> Streamlit dashboard (MVP)
      +--> FastAPI endpoints (planned)
                     |
              AgentSentinel core
       +-------------+-----------------------+
       |             |                       |
  Monitoring,     Evaluation,           Diagnostics and
  detection       benchmarking          actionable insights
       +-------------+-----------------------+
                     ^
                     | normalized execution events
             Integration Layer (planned)
       +-------------+-----------------------+
       |                                     |
  Framework adapters                   Provider gateway/adapters
  LangGraph (MVP)                      LiteLLM (MVP)
  LangChain, CrewAI,                   Groq (MVP)
  AutoGen, custom agents               OpenAI, Gemini, Claude,
                                       Ollama, compatible providers
       |                                     |
       +---------- Agent execution -----------+
                     |
         LangSmith tracing/debugging
         (initial observability integration)
```

This is the target architecture, not a claim that these integrations or services already exist. The initial MVP will prove the integration boundary with a LangGraph sample agent and Groq through LiteLLM. Adapters for additional providers and frameworks are future expansion. LangSmith supplies traces for debugging; AgentSentinel's product analysis consumes normalized events and must continue to work independently.

### Architecture boundaries

- **Integration Layer:** The boundary between external AI systems and AgentSentinel. It will contain LLM provider adapters, agent framework adapters, and normalized trace-event schemas. Adapters translate source-specific details into those events; downstream product features should not need provider- or framework-specific branches.
- **LiteLLM:** The initial unified LLM gateway, used to standardize model access across the providers it supports. Groq is the MVP provider; other compatible providers are future integrations.
- **LangSmith:** The primary tracing and debugging platform during initial development. It is an observability integration, not the entire product or the source of truth for AgentSentinel's custom decisions.
- **AgentSentinel core:** Owns monitoring signals, failure definitions, evaluation criteria, diagnostics, and future benchmarks. These features operate on normalized events and must be usable and testable without querying LangSmith.
- **Agent frameworks:** LangGraph is used for the MVP. LangChain, CrewAI, AutoGen, and custom Python agents are future framework integrations.
- **FastAPI** and **Streamlit** are separate access surfaces. The API should expose application capabilities; the dashboard should present them to a person.
- Provider- and framework-specific details should stay at integration boundaries so core application logic can be tested without live services or credentials.

## Agent execution workflow

The intended workflow is:

1. **Receive a request.** Validate the input and assign an execution identifier.
2. **Prepare execution context.** Establish the run state and any safe metadata needed for correlation.
3. **Run an agent.** The selected framework executes its workflow. In the MVP, this is a LangGraph sample agent with calculator, knowledge-base, and mock-weather tools.
4. **Call a model.** The agent uses LiteLLM as the unified gateway; Groq is the initial model provider.
5. **Adapt and normalize.** Framework and provider integrations map source-specific observations to AgentSentinel's normalized execution-event schema.
6. **Trace for debugging.** LangSmith is used for tracing and debugging during initial development. This observability path is separate from the normalized-event path used by product features.
7. **Monitor and detect.** AgentSentinel applies its own rules to normalized events, for example identifying exceptions, invalid output, timeouts, or configured threshold violations.
8. **Evaluate and diagnose.** AgentSentinel applies explicit evaluation criteria and combines the results with run evidence to provide bounded, actionable explanations. Do not treat a model-generated explanation as proof.
9. **Present insights.** The MVP Streamlit dashboard displays runs and findings. FastAPI remains a planned access surface, not a required MVP deliverable.
10. **Test and improve.** Tests verify normalization and application-owned behavior; developers use AgentSentinel insights and LangSmith traces for their distinct purposes.

## Responsibilities of each module

These are intended responsibilities; the current scaffold contains no application logic. The Integration Layer is part of the proposed architecture and may be added as an `app/integrations/` package when implementation begins; that package is not present in the current scaffold.

| Path | Responsibility |
|---|---|
| `app/` | Top-level Python application package and shared application assembly. |
| `app/agent/` | MVP LangGraph sample agent and its workflow state. Future framework adapters should keep framework-specific details out of core analysis. |
| `app/integrations/` (planned) | Provider and framework adapters, LiteLLM gateway integration, LangSmith observability integration, and normalized trace-event schemas. |
| `app/monitoring/` | AgentSentinel-owned run records, metrics, and monitoring interfaces operating on normalized events. |
| `app/detection/` | Rules and classifiers that identify failure conditions from normalized events. |
| `app/evaluation/` | Evaluation criteria, evaluators, structured results, and future cross-model benchmarks. |
| `app/diagnostics/` | Converts monitoring, detection, and evaluation evidence into useful incident explanations and next steps. |
| `app/api/` | FastAPI application and HTTP routes for exposing project capabilities. |
| `app/dashboard/` | MVP Streamlit user interface for exploring executions and insights. |
| `core/` | Cross-cutting configuration, shared types, and foundational utilities. Keep this package small; avoid turning it into a miscellaneous catch-all. |
| `tests/` | pytest tests, organized to mirror the application modules where useful. |
| `data/` | Local development or sample data. Do not store credentials or unapproved sensitive production data here. |
| `knowledge_base/` | Future curated reference material used by diagnostic or evaluation features, if required. |
| `docs/` | Project documentation, design decisions, and usage guides. |

## Technology stack and purpose

| Technology | Planned purpose |
|---|---|
| Python 3.11 | Main implementation language and runtime. |
| Conda | Reuse the existing environment at `D:\AgentOps\venv`; do not create a second environment for this project. |
| LangGraph | MVP framework for defining and executing the sample stateful agent workflow. |
| Groq | MVP LLM provider. Access credentials only through protected configuration. |
| LiteLLM | Initial unified LLM gateway to standardize model access across supported providers. |
| LangSmith | Primary tracing and debugging platform during initial development; an observability integration independent of custom analysis. |
| Pydantic | Validate structured inputs, outputs, configuration, and application data. |
| FastAPI | Expose application functionality as HTTP endpoints. |
| Streamlit | Build a lightweight interactive dashboard for developers. |
| pytest | Automate unit and integration tests for application-owned behavior. |

### Planned integration coverage

The MVP uses **Groq** through **LiteLLM** and a **LangGraph** sample agent. Future provider integrations include OpenAI, Google Gemini, Anthropic Claude, Ollama, and other compatible providers. Future framework integrations include LangChain, CrewAI, AutoGen, and custom Python agents. These are expansion goals, not current implementations.

Langfuse is not part of this project. Qdrant, Redis, and Kafka are also out of scope unless a future requirement justifies revisiting the architecture.

## Two-day development roadmap

This is a deliberately constrained two-day MVP plan, not a promise to implement every future integration or production-scale capability. The goal is one end-to-end example that proves the provider/framework boundary and the independent analysis path.

### Day 1 — Build the integrated sample agent

- Define only the minimum normalized event and run shapes needed by the MVP.
- Implement one small LangGraph sample agent with calculator, knowledge-base, and mock-weather tools.
- Route its Groq model calls through LiteLLM.
- Connect LangSmith tracing for debugging, keeping it outside the custom analysis dependency path.
- Add focused tests for event normalization and key workflow success and failure cases; mock external calls where practical.

### Day 2 — Deliver independent insights and a dashboard

- Implement a small set of custom monitoring signals and failure categories against normalized events.
- Add focused evaluation and evidence-based diagnostic logic with pytest coverage.
- Build a minimal Streamlit dashboard for sample runs and actionable findings.
- Review privacy, explicit error handling, and independence of product analysis from LangSmith.

Keep this first slice intentionally narrow: one framework, one provider, three simple tools, and a small set of checks. Defer additional frameworks and providers, cross-model benchmarking, custom integrations, production-scale persistence, and a full API until the MVP demonstrates the normalized-event architecture.

## Engineering principles

- Prefer small, explicit modules and clear data contracts.
- Keep policy and business decisions deterministic and independently testable where practical.
- Separate orchestration, integrations, normalized event contracts, analysis, persistence, API, and presentation concerns.
- Keep provider- and framework-specific translation in adapters; do not leak those assumptions into core monitoring or evaluation.
- Handle errors explicitly; do not silently turn failures into success-shaped results.
- Test boundary conditions and failure paths, not just the happy path.
- Use dependency injection or small interfaces at external-service boundaries to make tests independent of live services.
- Add dependencies and infrastructure only to solve a current, concrete need.
- Document design decisions and update setup instructions when workflows change.
- Treat AI-generated diagnoses as assistance for human investigation, not authoritative facts.

## Security practices

- Never commit real API keys, tokens, passwords, or private connection strings.
- Use environment variables or a suitable secret manager for credentials. `.env.example` contains names only; the real `.env` file must remain local and ignored by Git.
- Do not print secrets, include them in exceptions, or attach them to traces.
- Send only the minimum necessary input and metadata to external model and tracing services.
- Decide what prompt, output, and tool data may be retained before enabling tracing; redact or omit sensitive fields.
- Validate API inputs and avoid exposing internal stack traces or credentials in API responses.
- Use least-privilege credentials, rotate exposed keys, and review staged changes before committing.
- Do not put virtual-environment contents or private data in application source folders.

## Current development status

- **Status:** foundation/scaffolding stage.
- The project package layout and documentation are being established.
- No application logic is intended in the initial scaffold.
- AgentSentinel is designed to be provider-agnostic and framework-extensible, with normalized execution events as the boundary to its core features.
- The MVP is planned around a LangGraph sample agent, Groq through LiteLLM, calculator/knowledge-base/mock-weather tools, LangSmith tracing, custom monitoring and failure detection, evaluation, diagnostics, and a Streamlit dashboard.
- LangSmith is the primary tracing/debugging integration for initial development, but is not yet integrated and must not own custom monitoring, failure detection, evaluation, or diagnostics.
- Additional providers and frameworks, cross-model benchmarking, and custom integrations are future expansion goals.
- The existing Conda environment is at `D:\AgentOps\venv`; it must be reused, not recreated or modified as part of the scaffold.
- Dependency installation instructions are an initial draft, not a validated setup procedure; do not assume the declared dependencies have been installed or tested as a complete environment.
- No Langfuse, Qdrant, Redis, or Kafka integration is planned.

## Instructions for future AI coding assistants

1. Read this file and `README.md` before proposing or implementing project changes.
2. Inspect the workspace and report conflicts before overwriting existing files.
3. Preserve the existing environment at `D:\AgentOps\venv`. Never create, replace, delete, or move it.
4. Never read or reveal `.env`; do not modify it. Keep `.env.example` limited to empty placeholders.
5. Do not modify `.gitignore` or `requirements.txt` unless the user explicitly authorizes that change.
6. Do not install packages, run external services, or make network calls unless the user asks.
7. Preserve the provider-agnostic, framework-extensible architecture: core features consume normalized execution events, not provider- or framework-specific traces.
8. Use LiteLLM as the initial unified LLM gateway; Groq and LangGraph are MVP choices, not permanent architectural assumptions.
9. Use LangSmith as the primary tracing and debugging integration during initial development. Keep custom monitoring, failure detection, evaluation, and diagnostics independent of LangSmith availability or APIs.
10. Do not add Langfuse. Do not add Qdrant, Redis, or Kafka without a clear, approved requirement.
11. Keep changes scoped, avoid unnecessary files and abstractions, and do not add application logic during a scaffold-only task.
12. Add or update tests when implementing behavior, and report which checks were actually run.
13. Explain important decisions in beginner-friendly language and distinguish implemented behavior from planned behavior.
