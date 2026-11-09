<div align="center">

# AgentSentinel

**Monitor. Evaluate. Diagnose. Improve.**

An AI agent observability and evaluation platform that records what an agent actually did during each run, detects failures, scores response quality, and produces diagnostic reports.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/ORM-SQLAlchemy-D71F00)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)
![Alembic](https://img.shields.io/badge/Migrations-Alembic-6BA81E)
![LiteLLM](https://img.shields.io/badge/Providers-LiteLLM-7C3AED)
![Tests](https://img.shields.io/badge/Tests-Automated-success)

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Capabilities](#2-key-capabilities)
3. [System Architecture](#3-system-architecture)
4. [Agent Execution Flow](#4-agent-execution-flow)
5. [Monitoring and Event Lifecycle](#5-monitoring-and-event-lifecycle)
6. [Failure Detection](#6-failure-detection)
7. [Evaluation Engine](#7-evaluation-engine)
8. [Diagnostic Engine](#8-diagnostic-engine)
9. [Persistent Storage](#9-persistent-storage)
10. [FastAPI Backend](#10-fastapi-backend)
11. [Streamlit Dashboard](#11-streamlit-dashboard)
12. [Screenshot Gallery](#12-screenshot-gallery)
13. [Development Journey](#13-development-journey)
14. [Technology Stack](#14-technology-stack)
15. [Getting Started](#15-getting-started)
16. [Design Principles and Limitations](#16-design-principles-and-limitations)
17. [Author](#17-author)

---

## 1. Overview

A typical AI chatbot receives a question and returns an answer. When that answer is wrong, slow, or expensive, there is little visibility into why.

AgentSentinel is a monitoring and quality-assurance layer for AI agents. Instead of observing only the final answer, it captures the full execution lifecycle and uses it to answer questions such as:

| Question | Where AgentSentinel answers it |
|---|---|
| Did the agent run succeed? | Execution status, Failures page |
| Which provider and model were used? | `llm_call` events |
| How many LLM calls were made? | Per-run LLM call table |
| Which tools were executed? | `tool_call` events |
| How long did the run take? | Run and LLM latency |
| How many tokens were consumed? | Provider-reported token usage |
| Did an LLM or tool call fail? | Failure detector |
| Was the response correct and relevant? | Evaluation engine |
| Did the response include unsupported claims? | Hallucination indicators |
| What should be investigated first? | Diagnostic reports |

The goal is to make agent behavior **observable, measurable, and easier to diagnose**.

---

## 2. Key Capabilities

- **LangGraph agent** with a calculator, a local knowledge-base search, and a simulated weather tool.
- **Provider-agnostic model factory** supporting Groq (ChatGroq) and OpenAI, Anthropic, and Gemini (via LiteLLM).
- **Standardized monitoring events**: `run_started`, `llm_call`, `tool_call`, `run_completed`, `run_failed`.
- **Token and latency analytics** based on provider-reported usage, never fabricated.
- **Failure detection** over recorded execution events.
- **Two evaluators**: a rule-based heuristic evaluator and an LLM judge.
- **Rule-based diagnostic engine** producing findings with a Healthy, Warning, or Critical status.
- **Persistence** with SQLAlchemy, SQLite, repository classes, and Alembic migrations.
- **FastAPI REST backend** and a **six-page Streamlit dashboard**.

---

## 3. System Architecture

### Diagram 1: Complete System Architecture

```mermaid
flowchart TB
    User([User])

    subgraph UI["Presentation Layer"]
        ST["Streamlit Dashboard<br/>Playground, Overview, Executions,<br/>Failures, Evaluations, Diagnostics"]
    end

    subgraph API["API Layer"]
        FA["FastAPI Backend<br/>REST endpoints"]
    end

    subgraph AGENT["Agent Layer"]
        LG["LangGraph Agent"]
        MF["Model Factory"]
        subgraph TOOLS["Agent Tools"]
            T1["Calculator<br/>AST-based"]
            T2["Knowledge Base Search<br/>Markdown and text"]
            T3["Weather<br/>Simulated"]
        end
    end

    subgraph PROV["LLM Providers"]
        P1["Groq<br/>ChatGroq"]
        P2["OpenAI<br/>LiteLLM"]
        P3["Anthropic<br/>LiteLLM"]
        P4["Gemini<br/>LiteLLM"]
    end

    subgraph MON["Monitoring Layer"]
        EL["EventLogger<br/>in-memory capture and redaction"]
        ES["EventSink"]
    end

    subgraph ANALYSIS["Analysis Layer"]
        FD["Failure Detector"]
        EV["Evaluation Engine<br/>RuleBasedEvaluator / LLMJudge"]
        DG["Diagnostic Engine<br/>RuleBasedDiagnosticEngine"]
    end

    subgraph STORE["Persistence Layer"]
        SQS["SQLAlchemyEventSink"]
        REPO["Repositories<br/>ExecutionEvent / EvaluationResult / DiagnosticReport"]
        ORM["SQLAlchemy ORM"]
        DB[("SQLite Database<br/>Alembic migrations")]
    end

    User --> ST
    ST -->|"HTTP request"| FA
    FA -->|"run agent"| LG
    LG --> MF
    MF --> P1 & P2 & P3 & P4
    LG <-->|"tool calls and results"| TOOLS

    LG -->|"lifecycle events"| EL
    EL --> ES
    ES --> SQS
    SQS --> REPO
    REPO --> ORM
    ORM --> DB

    FA -->|"evaluation request"| EV
    FA -->|"diagnostic request"| DG
    FA -->|"failure analysis"| FD
    EV --> REPO
    DG --> REPO
    FD -.->|"reads events"| REPO
    EV -.->|"reads events"| REPO
    DG -.->|"reads events and evaluations"| REPO

    FA -->|"JSON responses"| ST
    ST -->|"visualization"| User
```

---

## 4. Agent Execution Flow

### Diagram 2: Complete Agent Execution Flow

```mermaid
flowchart TD
    A([User enters a question]) --> B["Streamlit sends POST request<br/>to the agent run endpoint"]
    B --> C{"FastAPI validates<br/>the question"}
    C -->|"Invalid or empty"| C1["Return validation error<br/>to dashboard"]
    C -->|"Valid"| D["Agent execution begins"]
    D --> E["Assign Run ID"]
    E --> F["Select provider and model<br/>via Model Factory"]
    F --> G["Emit run_started event"]
    G --> H["LangGraph invokes the LLM"]
    H --> I{"LLM call<br/>successful?"}
    I -->|"No"| I1["Capture llm_call event<br/>status: failure, with error"]
    I1 --> Z1["Emit run_failed event"]
    I -->|"Yes"| J["Capture llm_call event<br/>duration and token usage"]
    J --> K{"Tool required?"}
    K -->|"Yes"| L["Execute ToolNode"]
    L --> M{"Tool call<br/>successful?"}
    M -->|"No"| M1["Capture tool_call event<br/>status: failure, with error"]
    M1 --> Z1
    M -->|"Yes"| N["Capture tool_call event<br/>tool name and duration"]
    N --> O["Send tool results<br/>back to the LLM"]
    O --> H
    K -->|"No, final response ready"| P["Produce final response"]
    P --> Q["Emit run_completed event<br/>total execution latency"]
    Q --> R["Persist execution events"]
    Z1 --> R
    R --> S{"Run outcome"}
    S -->|"Success"| T["Return response and Run ID"]
    S -->|"Failure"| U["Return error information<br/>and Run ID"]
    T --> V["Dashboard displays response"]
    U --> V
    V --> W["Executions page shows<br/>event timeline, tokens and latency"]
    W --> X([Done])
```

---

## 5. Monitoring and Event Lifecycle

Monitoring is built into agent execution. Every run produces standardized events that describe what happened, in order.

| Event type | Emitted when | Typical contents |
|---|---|---|
| `run_started` | A run begins | Run ID, timestamp, input information |
| `llm_call` | An LLM invocation finishes | Provider, model, duration, token usage, status, error |
| `tool_call` | A tool finishes executing | Tool name, duration, status, input/output information, error |
| `run_completed` | The run succeeds | Total duration, final status |
| `run_failed` | The run fails | Error details, total duration |

Each event carries an event ID, run ID, timestamp, status, duration, input/output information, error details, and additional metadata. Sensitive values pass through the existing redaction logic before being stored.

### Diagram 3: Monitoring and Event Lifecycle

```mermaid
flowchart LR
    subgraph GEN["Event Generation During a Run"]
        direction TB
        RS["Run start"] --> E1["run_started"]
        LC["LLM call"] --> E2["llm_call<br/>provider, model, tokens, latency"]
        TC["Tool call"] --> E3["tool_call<br/>tool name, status, duration"]
        RC["Run completion"] --> E4["run_completed"]
        RF["Run failure"] --> E5["run_failed<br/>error details"]
    end

    E1 & E2 & E3 & E4 & E5 --> EL["EventLogger<br/>in-memory capture<br/>redaction applied"]
    EL --> ES{"EventSink<br/>configured?"}
    ES -->|"No"| MEM["In-memory events only"]
    ES -->|"Yes"| SQS["SQLAlchemyEventSink"]
    SQS --> REPO["ExecutionEventRepository"]
    REPO --> DB[("SQLite<br/>execution events table")]
    DB -->|"retrieved by run ID"| API["FastAPI<br/>executions and events endpoints"]
    API --> DASH["Dashboard<br/>Executions page"]
```

### Token usage and latency

Token usage is read from the provider-reported `AIMessage.usage_metadata` when it is available.

| Metric | Notes |
|---|---|
| Input / output / total tokens | Reported by the provider |
| Usage availability | Tracked explicitly per LLM call |
| LLM call latency | Duration of each model call |
| Total execution latency | Duration of the complete run |

When a provider does not report usage, the value is treated as **unavailable**, not zero. Estimated cost is shown as unavailable rather than fabricated.

---

## 6. Failure Detection

The `FailureDetector` analyzes recorded execution events and identifies where a run went wrong, rather than showing a generic error.

It examines:

- Overall run status.
- Failed LLM calls.
- Failed tool calls.
- Run-level failures.
- Execution duration.
- Available error information.

### Diagram 5: Failure Detection Workflow

```mermaid
flowchart TD
    A["Retrieve execution events"] --> B["Group events by Run ID"]
    B --> C["For each run: analyze run status"]
    C --> D{"run_failed event<br/>present?"}
    D -->|"Yes"| D1["Record run-level failure<br/>with error details"]
    D -->|"No"| E["Inspect LLM calls"]
    D1 --> E
    E --> F{"Any failed<br/>llm_call?"}
    F -->|"Yes"| F1["Record LLM failure<br/>provider, model, error"]
    F -->|"No"| G["Inspect tool calls"]
    F1 --> G
    G --> H{"Any failed<br/>tool_call?"}
    H -->|"Yes"| H1["Record tool failure<br/>tool name, error"]
    H -->|"No"| I["Calculate execution information<br/>duration and event counts"]
    H1 --> I
    I --> J{"Determine<br/>overall status"}
    J -->|"Failures recorded"| K["Status: failed"]
    J -->|"Completed without failures"| L["Status: healthy"]
    K --> M["Produce failure summary"]
    L --> M
    M --> N["Display on Failures page"]
```

---

## 7. Evaluation Engine

AgentSentinel provides two evaluators with different trade-offs.

| | `RuleBasedEvaluator` | `LLMJudge` |
|---|---|---|
| Approach | Lexical heuristics and keyword-oriented checks | A configured chat model with structured output validation |
| Cost and speed | Fast, no model call | Requires an LLM call |
| Determinism | Deterministic | Depends on the judge model |
| Metrics | Correctness, relevance, context relevance, hallucination indicators, overall score | Structured evaluation validated against a schema |
| Best for | Cheap, repeatable first-pass checks | Judgments that need language understanding |

> **Important:** the rule-based evaluator uses heuristics. It is a practical screening tool, not a semantic or scientifically validated evaluator. Treat its scores as indicators.

Component scores use a 0 to 1 scale. The overall score is shown on a 0 to 100 scale and compared against a pass/fail threshold.

### Diagram 4: Evaluation Workflow

```mermaid
flowchart TD
    A["Select an execution"] --> B["Retrieve execution events"]
    B --> C["Supply actual answer<br/>and reference information"]
    C --> D{"Choose evaluator"}
    D -->|"Rule-based"| E1["RuleBasedEvaluator<br/>lexical heuristics"]
    D -->|"LLM judge"| E2["LLMJudge<br/>chat model with structured output"]
    E2 --> E3{"Structured output<br/>valid?"}
    E3 -->|"No"| E4["Report evaluation error"]
    E3 -->|"Yes"| F
    E1 --> F["Calculate metrics<br/>correctness, relevance,<br/>context relevance, hallucination"]
    F --> G["Aggregate into overall score"]
    G --> H{"Score meets<br/>threshold?"}
    H -->|"Yes"| I1["Result: pass"]
    H -->|"No"| I2["Result: fail"]
    I1 --> J["Persist result<br/>EvaluationResultRepository"]
    I2 --> J
    J --> K["Display on Evaluations page"]
```

---

## 8. Diagnostic Engine

The `RuleBasedDiagnosticEngine` combines execution events and, when available, evaluation results to produce a diagnostic report. Reports help developers understand potential problems and decide what to investigate first.

| Detectable issue | Source |
|---|---|
| Insufficient execution data | Execution events |
| Failed agent run | Execution events |
| LLM failure | `llm_call` events |
| Tool failure | `tool_call` events |
| Low response relevance | Evaluation result |
| High hallucination score | Evaluation result |
| Low overall evaluation score | Evaluation result |
| Slow execution | Execution duration |

Each report contains a summary, individual findings with a severity, and an overall status: **Healthy**, **Warning**, or **Critical**.

### Diagram 6: Diagnostic Workflow

```mermaid
flowchart TD
    A["Select an execution"] --> B["Retrieve execution events"]
    B --> C{"Evaluation<br/>available?"}
    C -->|"Yes"| C1["Retrieve evaluation result"]
    C -->|"No"| C2["Continue with events only"]
    C1 --> D["Analyze execution health"]
    C2 --> D
    D --> E["Apply diagnostic rules"]
    E --> R1["Insufficient data?"]
    E --> R2["Run, LLM or tool failure?"]
    E --> R3["Low relevance or<br/>high hallucination?"]
    E --> R4["Low overall score?"]
    E --> R5["Slow execution?"]
    R1 & R2 & R3 & R4 & R5 --> F["Generate findings<br/>with severity"]
    F --> G{"Determine overall status"}
    G -->|"No findings"| H1["Healthy"]
    G -->|"Moderate findings"| H2["Warning"]
    G -->|"Severe findings"| H3["Critical"]
    H1 & H2 & H3 --> I["Persist report and findings<br/>DiagnosticReportRepository"]
    I --> J["Display findings on Diagnostics page"]
```

---

## 9. Persistent Storage

Execution data is preserved in SQLite through SQLAlchemy so it can be retrieved later by the API and dashboard.

| Stored data | Repository |
|---|---|
| Execution events | `ExecutionEventRepository` |
| Evaluation results | `EvaluationResultRepository` |
| Diagnostic reports and findings | `DiagnosticReportRepository` |

- `SQLAlchemyEventSink` connects the monitoring layer to storage, so the agent never touches the database directly.
- Repositories isolate query logic from the API and analysis engines.
- Alembic manages schema migrations.

### Diagram 7: Database ER Diagram

Key columns are shown. Execution events do not belong to a parent run table; evaluations and diagnostic reports are associated with an execution through its run ID. Findings belong to a diagnostic report.

```mermaid
erDiagram
    EXECUTION_EVENT {
        string event_id PK
        string run_id
        string event_type
        datetime timestamp
        string status
        string provider
        string model
        string tool_name
        float duration_ms
        json metadata
        string error
    }

    EVALUATION_RESULT {
        string id PK
        string run_id
        float overall_score
        boolean passed
        datetime created_at
    }

    DIAGNOSTIC_REPORT {
        string report_id PK
        string run_id
        string status
        string summary
        datetime timestamp
    }

    DIAGNOSTIC_FINDING {
        string id PK
        string report_id FK
        string severity
        string code
        string message
    }

    DIAGNOSTIC_REPORT ||--o{ DIAGNOSTIC_FINDING : "contains"
    EXECUTION_EVENT }o..o{ EVALUATION_RESULT : "associated by run_id"
    EXECUTION_EVENT }o..o{ DIAGNOSTIC_REPORT : "associated by run_id"
    EVALUATION_RESULT }o..o{ DIAGNOSTIC_REPORT : "associated by run_id"
```

> Column lists are a simplified view for readability. Refer to the SQLAlchemy models for the authoritative schema.

---

## 10. FastAPI Backend

The backend exposes agent execution, history, evaluation, and diagnostics. Interactive OpenAPI documentation is served by FastAPI at `/docs`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/agent/run` | Run the agent on a question |
| `GET` | `/api/v1/executions` | List executions |
| `GET` | `/api/v1/executions/{run_id}` | Get one execution |
| `GET` | `/api/v1/executions/{run_id}/events` | List events for an execution |

Evaluation and diagnostic report endpoints are grouped under the `evaluations` and `diagnostics` tags in the generated OpenAPI documentation, covering creation and retrieval.

---

## 11. Streamlit Dashboard

| Page | Purpose |
|---|---|
| **Agent Playground** | Submit a question, run the agent, and read the actual response. Each submission starts a new recorded run. |
| **Overview** | Platform health metrics: total, successful and failed executions, success rate, evaluation count and average score, diagnostic report count, with charts. |
| **Executions** | Run history and per-run details: event timeline, token usage, LLM call latency. |
| **Failures** | Runs that did not complete successfully. |
| **Evaluations** | Response evaluation results and scores. |
| **Diagnostics** | Diagnostic reports, status, and individual findings. |

The Failures page notes that the executions API does not provide status-filtered totals, so failure counts apply to the currently loaded page only.

---

## 12. Screenshot Gallery

> **Placeholders:** the images below are referenced by relative path and will render once the files are added to `docs/screenshots/`.

### Agent Playground
<img width="1917" height="865" alt="Screenshot 2026-11-10 022719" src="https://github.com/user-attachments/assets/acfcbf56-a484-4e80-9790-598f617ce377" />


### Overview
<img width="1917" height="890" alt="Screenshot 2026-11-10 022747" src="https://github.com/user-attachments/assets/eb5e9c2e-bec9-43f6-ba2b-ce70572d5ba3" />



### Executions
<img width="1917" height="782" alt="Screenshot 2026-11-10 025325" src="https://github.com/user-attachments/assets/77632b95-34af-40a7-80b4-bdc0f471b89c" />



### Failures
<img width="1650" height="805" alt="Screenshot 2026-11-10 022829" src="https://github.com/user-attachments/assets/90d173e7-b4b5-40ba-84bd-cca4fc028b70" />


### Evaluations
<img width="1917" height="781" alt="Screenshot 2026-11-10 024716" src="https://github.com/user-attachments/assets/fb422c0e-c440-4403-a8aa-319ac791702b" />


### Diagnostics
<img width="1897" height="876" alt="Screenshot 2026-11-10 022850" src="https://github.com/user-attachments/assets/6e9dcc58-586d-4bc8-8294-652013251b67" />


### Token Usage and Latency
<img width="1917" height="820" alt="Screenshot 2026-11-10 024838" src="https://github.com/user-attachments/assets/845ecbc7-0dbf-4291-a736-7d5fc08b8b8c" />


---

## 13. Development Journey

AgentSentinel was built in layers, where each phase gave the next one something concrete to build on.

| # | Phase | Builds on |
|---|---|---|
| 1 | Defined project vision and modular architecture | Foundation |
| 2 | Built the LangGraph agent and tools | Architecture |
| 3 | Introduced a provider-agnostic model factory | Agent |
| 4 | Implemented standardized monitoring events | Agent behavior worth observing |
| 5 | Integrated monitoring directly into agent execution | Events and agent |
| 6 | Built the failure detection engine | Recorded events |
| 7 | Developed the evaluation engine | Agent responses and events |
| 8 | Developed the diagnostic engine | Events and evaluations |
| 9 | Added SQLAlchemy repositories and persistent storage | Events, evaluations, diagnostics |
| 10 | Introduced Alembic migrations | Persistent schema |
| 11 | Developed the FastAPI backend | Agent, analysis, storage |
| 12 | Built the Streamlit dashboard | API |
| 13 | Added token usage and latency analytics | Monitoring, API, dashboard |
| 14 | Implemented security and reliability improvements | Whole system |
| 15 | Integrated components and performed automated testing | Whole system |

### Diagram 8: Development Journey

```mermaid
flowchart TD
    P1["1. Vision and modular architecture"] --> P2["2. LangGraph agent and tools"]
    P2 --> P3["3. Provider-agnostic model factory"]
    P3 --> P4["4. Standardized monitoring events"]
    P4 --> P5["5. Monitoring integrated into execution"]

    P5 --> P6["6. Failure detection engine"]
    P5 --> P7["7. Evaluation engine"]
    P6 --> P8["8. Diagnostic engine"]
    P7 --> P8

    P8 --> P9["9. SQLAlchemy repositories and storage"]
    P9 --> P10["10. Alembic migrations"]
    P10 --> P11["11. FastAPI backend"]
    P11 --> P12["12. Streamlit dashboard"]
    P12 --> P13["13. Token usage and latency analytics"]
    P13 --> P14["14. Security and reliability improvements"]
    P14 --> P15["15. Integration and automated testing"]

    P15 --> GOAL(["Complete platform:<br/>monitoring, evaluation, diagnostics,<br/>persistence, API and dashboard"])
```

---

## 14. Technology Stack

| Area | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM providers | Groq (ChatGroq); OpenAI, Anthropic, Gemini via LiteLLM |
| Backend | FastAPI |
| Dashboard | Streamlit |
| Persistence | SQLAlchemy, SQLite |
| Migrations | Alembic |
| Testing | Automated test suite |

---

## 15. Getting Started

> Adjust commands to match your repository layout and environment.

```bash
# 1. Clone and enter the project
git clone <your-repository-url>
cd AgentSentinel

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure provider credentials (example)
export GROQ_API_KEY="your-key"

# 4. Apply database migrations
alembic upgrade head

# 5. Start the backend (module path may differ in your project)
uvicorn app.main:app --reload

# 6. Start the dashboard (entry file may differ in your project)
streamlit run dashboard/app.py
```

The Streamlit dashboard runs on `http://localhost:8501` by default, and the backend's interactive API docs are available at `/docs`.

---

## 16. Design Principles and Limitations

<details>
<summary><strong>Engineering principles</strong></summary>

- **Observe the process, not only the answer.** Events describe the whole run.
- **Provider independence.** The monitoring system does not depend on a specific LLM vendor.
- **Honest metrics.** Unavailable token usage is shown as unavailable, and cost is not fabricated.
- **Separation of concerns.** Agent, monitoring, analysis, storage, API, and UI are separate modules.
- **Safe tooling.** The calculator evaluates expressions through an AST-based approach rather than arbitrary code execution.
- **Redaction.** Sensitive information is handled through redaction logic before events are stored.

</details>

<details>
<summary><strong>Known limitations</strong></summary>

- The rule-based evaluator relies on lexical heuristics and is not a semantic or scientifically validated evaluator.
- LLM-judge results depend on the configured judge model.
- The weather tool returns simulated data for selected cities.
- Failure counts on the dashboard apply to the loaded page, because the executions API does not provide status-filtered totals.
- Estimated cost is currently reported as unavailable.
- SQLite suits local and development use; larger deployments would need a different database configuration.

</details>

---

## 17. Author

<div align="center">

### Prithvi Raj Mukhiya

B.Tech, Electronics and Communication Engineering
Indian Institute of Information Technology, Kota (IIIT Kota)

</div>

**Areas of Interest**

- Generative AI
- AI Agents and Agentic AI
- Machine Learning
- Deep Learning
- Data Science
- MLOps

I enjoy building intelligent systems that are not only capable but also observable, measurable, and trustworthy, and I am keen to keep exploring emerging AI technologies and the engineering practices that make them reliable in production.
