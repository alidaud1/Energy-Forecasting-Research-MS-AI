# Execution Roadmap for a Multi‑Agent Transformer Framework for Joint Load & Solar Forecasting

## Executive summary

Your **base paper** proposes an **LLM-driven, multi-agent collaboration framework** that wraps the *entire load forecasting lifecycle*—task preparation → model search/training → deployment/postprocessing—into specialized agents that communicate through a **topic-based message routing mechanism**, enabling **human-in-the-loop** interaction throughout the pipeline. citeturn9view0turn10view0turn16view0

Your **proposal slides** reuse that same *core agentic workflow idea* but extend it to **renewables and joint decision-making** by explicitly adding a **Solar Forecasting Agent** and a **Forecast Integration Agent** (and conceptually separating “Feature Engineering” and “Evaluation” modules). Visually, your slides’ “existing vs proposed” highlight the two new solar/integration components as the key upgrade (i.e., moving beyond load-only). *(Evidence shown below from your slides.)*

The biggest practical takeaway for starting **today** is:  
You should **first implement the base paper’s agentic pipeline (minimal version)** on the **GEFCom load track** and **GEFCom solar track** separately, then **add the “Forecast Integration Agent”** that combines (Load forecast, Solar forecast) into a joint *net-load / joint-report* artifact. In parallel, you can introduce a transformer forecasting backbone (e.g., PatchTST / iTransformer) as one of the candidate models the “Model Developer” can train. PatchTST is built around patching and channel-independence for multivariate time series, and iTransformer emphasizes variate-centric attention; both are widely used transformer baselines for forecasting. citeturn13view0turn13view1

## Base paper core architecture and how it differs from your proposal

### What the base paper actually builds

At the system level, the base paper defines **five specialized agents** plus a visualization interface:

- **Task Manager** (the only agent that talks to the human; coordinates stages). citeturn9view0turn10view2  
- **Preparation Assistant** (collects task metadata, configures metrics, runs preprocessing). citeturn9view0turn10view2  
- **Model Manager** (plans model/hyperparameter exploration using an *interactive Bayesian optimization* strategy that can incorporate user guidance). citeturn10view0turn16view0  
- **Model Developer** (executes training + evaluation tool calls; modular “dispatcher” separating planning from execution). citeturn16view0  
- **Deployment Operator** (runs the chosen model on new data and supports **postprocessing** operations like manual overrides and scaling rules). citeturn10view2turn10view3  

Two design details matter for implementation:

The agents communicate via a **subscription-based message routing mechanism**: each agent registers for topics, messages are published to a message pool with a topic tag, and agents consume only topics they subscribe to. citeturn9view0turn16view0  

The Model Manager’s search loop is **interactive Bayesian optimization**: user/LLM guidance can prune the search space or impose preferences before Optuna-like selection of the next batch. The paper explicitly describes this as an “interactive Bayesian optimization algorithm” integrated with user guidance. citeturn10view0turn10view1  

The paper also states it leverages **AutoGen** for multi-agent message routing/management and uses **Optuna** as the Bayesian optimization package. citeturn10view1turn11view0  

### What your proposal plans instead

Your proposal slides preserve the same *conceptual skeleton* (human user → task manager → preparation/modeling/deployment), but the plan differs in three concrete ways:

![Your slide: “Seven Specialized Agent Modules”](sandbox:/mnt/data/converted/slides/slide-09.png)

First, your proposal explicitly defines **seven specialized modules** (not five). In addition to task management and data preparation, it separates **Feature Engineering** and **Evaluation** as modules and adds **Load Forecasting**, **Solar Forecasting**, and **Forecast Integration** as distinct components.

![Your slide: “Existing vs Proposed” highlighting new agents](sandbox:/mnt/data/converted/slides/slide-18.png)

Second, compared to the base paper’s load-only scope, your proposal adds two **new components** as the key innovation:
- A **Solar Forecasting Agent** (solar power forecasting; weather–PV relationships; solar-specific tuning).  
- A **Forecast Integration Agent** (combine load + solar outputs; resolve inconsistencies; produce “final joint forecasts” and a decision-ready summary).

Third, your methodology slide lists a **fixed set of baseline models** (MLR, SVR, XGBoost, MLP, LSTM, GRU, CNN) and **datasets** (GEFCom2014-L_V2, GEFCom2014-S_V2). citeturn17search1turn3search4  
The base paper’s modeling stage is more explicitly an **interactive AutoML/search** problem—searching over model families + features + hyperparameters—with user steerability, rather than a static shortlist. citeturn10view0turn16view0  

### Direct comparison table

| Dimension | Base paper (what you are building upon) | Your proposal slides (your planned thesis system) |
|---|---|---|
| Forecasting scope | **Load forecasting only**, with human-in-loop interaction across pipeline stages. citeturn5view0turn9view0 | **Load + solar**, with explicit “joint” integration and renewable focus (solar agent + integration agent). *(shown in your slides above)* |
| Agents/modules | 5 core agents: Task Manager, Preparation Assistant, Model Manager, Model Developer, Deployment Operator, plus visualization. citeturn9view0turn16view0 | 7 modules: Task Manager, Data Preparation, Feature Engineering, Load Forecasting, Solar Forecasting, Forecast Integration, Evaluation. *(slide-09)* |
| Communication | Explicit **topic-based publish/subscribe** message pool. citeturn9view0turn16view0 | Not specified as a protocol; shown as a structural diagram of components (you can inherit base paper’s protocol). |
| Model selection | **Interactive Bayesian optimization**; user guidance can prune/preferences; Model Manager uses ReAct-like think→act loop. citeturn10view0turn16view0turn15view0 | Baseline model list is given; no explicit interactive BO loop shown (but you can adopt it). |
| Deployment | Explicit postprocessing strategies (manual override, time-based scaling, threshold-based scaling). citeturn10view3turn10view2 | Proposal emphasizes integration + evaluation; postprocessing isn’t emphasized (but you can reuse base paper’s deployment concepts). |

## Data and environment setup

### Your dataset archives and what they contain

You uploaded **two** archives (not one):  
- `GEFCom2014-L_V2.zip` (load forecasting track)  
- `GEFCom2014-S_V2.zip` (solar forecasting track)

These are competition-grade datasets designed for **probabilistic forecasting** (99 quantiles per horizon), but your proposal’s metrics (MAE/RMSE/MAPE) suggest you’ll likely start with **point forecasting** and later optionally extend to quantiles.

**Load dataset context (from primary sources):**  
GEFCom2014 load track uses hourly load data (2005–2010) and hourly temperature data (2001–2010) from **25 weather stations**, released incrementally in a rolling competition format; US federal holidays can be used. citeturn17search0turn17search1  

**Solar dataset context (from primary sources):**  
GEFCom2014 solar track forecasts hourly power for **three solar plants** (“zones”), using **ECMWF numerical weather prediction** variables; solar data pre-processing often involves handling **accumulated meteorological fields** (decumulation/differencing) and time feature extraction. citeturn3search1turn3search3turn3search4  

### What preprocessing is required for transformer-ready modeling

Transformers need data shaped into consistent tensors, typically:

- Input window: \([B, W, D]\)  
- Forecast horizon: \([B, H, 1]\) or \([B, H, T]\) for multi-task outputs  
- Known future covariates (optional): \([B, H, D_f]\)

Below are the **start-here preprocessing steps** per dataset.

#### Load track preprocessing checklist

The load track is “one load series + temperature stations” in a competition format. citeturn17search0turn17search1

1) **Parse the timestamp robustly**  
GEFCom load data commonly encodes date as an integer-like string and hour as `H:00`. You must convert it into a true `datetime` index before anything else.

2) **Remove records where the load target is missing**  
The load dataset contains temperature history that begins earlier than load history (temperature starts 2001; load starts 2005 in the standard track description). citeturn17search0turn17search1  
So you will see many rows where temperatures exist but load is blank. Those cannot be used as supervised targets.

3) **Consolidate weather stations into usable features**  
You have 25 stations. Options:
- Simple: mean / median temperature across stations (fast baseline).
- Better: learn a station-weighted temperature embedding.
- Literature-aligned: station selection is a known issue in GEFCom load modeling. citeturn17search1turn17search22  

4) **Create calendar features**  
Day-of-week, hour-of-day, month, weekend. The load track explicitly allows holiday calendar use. citeturn17search0turn17search1  

5) **Scaling**  
Fit scalers **only on training period**:
- `LOAD`: standard scale or robust scale.
- temperatures: standard scale.

6) **Windowing into transformer tensors**  
Choose:
- Lookback window \(W\): e.g., 168 hours (1 week) or 336 (2 weeks)
- Horizon \(H\): 24 hours for day-ahead (recommended for thesis consistency)

Output:
- `x_hist`: [W, D] includes past load + past temps + time features
- `x_future_known`: [H, Dk] includes known future time features, and (if allowed) future temperatures (in competition, future temp is uncertain; you can either (a) use oracle temp for offline evaluation, or (b) train a temp forecaster like classical GEFCom solutions do). citeturn17search1  

#### Solar track preprocessing checklist

Solar track typically includes separate `train*.csv` (target `POWER`) and `predictors*.csv` (meteorological features) for each task. citeturn3search3turn3search4  

1) **Merge targets with predictors**  
Join on `(ZONEID, TIMESTAMP)`. Use the overlap period for training.

2) **Decumulate the accumulated fields**  
Multiple sources describing the solar track note that several NWP fields are **accumulated**, and preprocessing differentiates them to obtain per-hour values. citeturn3search1turn3search3  
Practically:
- For each zone, sort by time.
- For each accumulated variable, compute `diff` within each daily forecast cycle; handle resets (often at 00:00).

3) **Handle nighttime and near-zero targets carefully**  
Solar output is normalized and has many zeros at night; MAE/RMSE are fine, but MAPE can explode. Solar papers often evaluate under hourly frameworks and may restrict to non-zero feasible windows for some steps. citeturn3search1turn3search3  

4) **Time features**  
Hour-of-day is extremely important for solar; many approaches segment by hour-of-day or add time encodings. citeturn3search3turn3search1  

5) **Scaling**  
Scale predictors by training statistics; target already normalized but still can be standardized.

6) **Windowing**  
Use the same (W, H) as load for consistency (e.g., W=168, H=24), but expect solar to benefit from weather covariates strongly.

### Python environment recommendation

You want reproducibility + compatibility with modern deep learning + agent orchestration. The base paper explicitly uses **AutoGen** for agent communication and **Optuna** for Bayesian optimization. citeturn10view1turn11view0  

Recommended environment (thesis-friendly, stable):

- Python: **3.10 or 3.11** (broad PyTorch ecosystem compatibility)
- Core data:
  - `numpy`, `pandas`, `scipy`
  - `pyarrow` (fast IO)
- ML baselines:
  - `scikit-learn`
  - `xgboost` (and optionally `lightgbm`)
- Deep learning:
  - `torch` (PyTorch), `torchvision` (optional)
  - `pytorch-lightning` (optional but helpful)
- Experiment tooling:
  - `optuna` (HPO; matches base paper’s tuning approach) citeturn10view1turn13view0  
  - `mlflow` or `wandb` (pick one)
  - `hydra-core` (config management)
- Agentic framework:
  - `pyautogen` / AutoGen package (multi-agent orchestration) citeturn11view0turn10view1  
- Plotting:
  - `matplotlib`, `seaborn` (optional), `plotly` (optional)
- Quality:
  - `black`, `ruff`, `mypy`, `pytest`

If you will implement transformer baselines:
- PatchTST paper is a strong backbone because it reduces attention cost via patching and uses channel-independence. citeturn13view0  
- iTransformer is a strong multivariate backbone due to variate-centric tokenization. citeturn13view1  

## Coding roadmap with phased execution

You said you’re confused about where to start. This is the exact sequence I’d direct as your research lead.

### Phase A: Data pipeline

**Goal:** one command produces clean, windowed datasets for both tracks.

Deliverables:
- `data/raw/` contains extracted CSVs
- `data/processed/` contains parquet (or cached arrays)
- a `Dataset` class that yields `(x_hist, x_future, y)` for load and solar
- unit tests for timestamp parsing and decumulation

Core steps:

1) Implement loaders
- `load_track_loader.py`: reads Load Task 1 (for full history) and/or selected tasks; outputs a single long dataframe with datetime index.
- `solar_track_loader.py`: reads Solar Task 15 (largest training span) + predictors; merges into one dataframe per zone.

2) Implement transformers-ready windowing
- `windowing.py`: generate sliding windows per zone with configurable `(W, H, stride)`.

3) Implement solar decumulation
- `solar_features.py`: for accumulated variables, compute *per-hour increments* with reset handling. This is essential per GEFCom solar preprocessing literature. citeturn3search1turn3search3  

4) Implement scaling properly
- Fit scalers on train split only; apply to val/test.

5) Create time series splits
- Use chronological splits, e.g. last 20% test, previous 10% validation.
- Optionally implement rolling-origin evaluation later.

### Phase B: Baseline model

**Goal:** reproduce a “simple version” of the base paper’s idea: agentic orchestration of forecasting workflow—without yet doing your full solar integration.

Deliverables:
- A minimal multi-agent runner that:
  - collects task configuration
  - trains a baseline model
  - produces an evaluation report
- Baseline models working end-to-end:
  - Linear regression / XGBoost
  - LSTM (optional)
- Logged experiment artifacts: metrics + plots

What to implement (minimal base paper replication):

1) Minimal agents (3–4 agents only)
- **TaskManagerAgent**: accepts user config and triggers the pipeline (mirrors base paper Task Manager role). citeturn9view0turn16view0  
- **PreparationAgent**: runs preprocessing, checks columns and splits (mirrors Preparation Assistant). citeturn9view0turn10view2  
- **ModelManagerAgent**: selects model family and hyperparameters; start with a simple ruleset, then optionally add Optuna. Base paper’s manager is ReAct-like and can incorporate user guidance. citeturn16view0turn15view0  
- **ModelDeveloperAgent**: executes training + evaluation calls; logs results (mirrors base paper Model Developer). citeturn16view0  

2) Communication mechanism
Implement the base paper’s publish/subscribe concept:
- Topics like: `chat`, `prepare`, `train`, `evaluate`, `deploy`
- A message pool (in-memory + persisted JSONL)

This matches the paper’s “message pool + topics + subscription” routing and lets you scale to more agents later. citeturn9view0turn16view0  

3) Model training stage logic
Start with:
- Fixed baseline configs (linear regression, XGBoost)
Then extend to:
- Optuna tuning (“interactive Bayesian optimization” in spirit). The base paper explicitly frames model search as Bayesian optimization with external guidance and also notes Optuna usage. citeturn10view0turn10view1turn17search1  

### Phase C: Full agentic framework + transformer integration

**Goal:** implement the **proposal’s extended agent set** and integrate transformer backbones.

Deliverables:
- 7-agent system (as per your slides)
- Transformer forecasting models (PatchTST or iTransformer) integrated as candidates
- Forecast Integration Agent that outputs:
  - joint report
  - net-load derived forecast (if definable)
  - confidence/probabilistic bands (optional)

How to build it systematically:

1) Add the solar pipeline as first-class
- Add a **SolarForecastingAgent** that can:
  - train models on solar track
  - tune solar-specific preprocessing (decumulation on/off; hour embeddings; etc.)
This matches your proposal and the known solar preprocessing needs. citeturn3search3turn3search1  

2) Add the ForecastIntegrationAgent (your key novelty)
- Inputs:
  - Load forecast \(\hat{L}_{t:t+H}\)
  - Solar forecast \(\hat{S}_{t:t+H}\)
- Outputs:
  - Joint artifacts: (load, solar, net-load) where net-load = load − solar if the same system/time alignment exists, or else a “decision-ready summary” as your slides indicate.

3) Add transformer forecasters
Start with one transformer (keep it thesis-feasible):
- PatchTST (good first choice; explicitly designed for multivariate forecasting with patching and channel-independence). citeturn13view0  
Then optionally:
- iTransformer (variate-centric attention for multivariate correlations). citeturn13view1  

4) Where “multi-agent” meets “transformer”
- Agentic AI layer decides training plans, runs experiments, and produces reports.
- Transformer is the forecasting core.
This is aligned with the base paper’s principle: LLMs as workflow orchestrators rather than only direct predictors. citeturn9view0turn16view0  

## Parallel writing strategy so thesis writing becomes easy

You should treat writing as a **running lab notebook + thesis skeleton**.

### What to write while coding the Methodology chapter

Write these subsections incrementally while implementing each module:

Data section (write during Phase A)
- Dataset description: GEFCom load track time ranges, 25 weather stations, competition incremental release context. citeturn17search0turn17search1  
- Solar track: 3 zones, ECMWF predictors, accumulated fields and need for decumulation. citeturn3search1turn3search3turn3search4  
- Your preprocessing: timestamp parsing, missing target handling, scaling, windowing parameters.

System architecture section (write during Phase B/C)
- Agent definitions: role, inputs/outputs, allowed tools (the base paper explicitly structures roles and tool-call actions). citeturn9view0turn16view0  
- Communication: publish/subscribe topics and message pool, directly mapped to the base paper mechanism. citeturn9view0turn16view0  
- Model search strategy: explain your tuning approach and how user guidance can enter (even if you start simple).

Modeling section (write as you add models)
- Baseline models: definitions and hyperparameters
- Transformer architecture: which backbone (PatchTST / iTransformer), window sizes, loss functions.
- Training protocol: optimizer, early stopping, seeds, compute.

Evaluation section (write as you produce plots)
- Metrics (MAE/RMSE; note pitfalls of MAPE for solar zeros)
- Train/val/test splits and why time-ordered
- Ablations: decumulation on/off; time features on/off; transformer vs traditional.

### A concrete documentation workflow

Create a `docs/` folder in your repo with:
- `docs/methodology_live.md` (updated weekly)
- `docs/exp_log.md` (every run: dataset, config hash, metrics)
- `docs/figures/` (saved with experiment IDs)
- `docs/agent_prompts/` (versioned prompts + tool schemas)

This mirrors the base paper’s emphasis on stage-aware visualization and logging, but keeps it thesis-ready. citeturn9view0turn16view0  

## Innovation check: the novelty you introduce vs the base paper

Based on the base paper’s stated scope and architecture, your proposal’s novelty is strongest in these gaps:

First, the base paper targets **interactive load forecasting** and defines a 5-agent workflow around load only. citeturn5view0turn9view0  
Your proposal explicitly adds:
- **Solar forecasting** as a first-class agent responsibility
- **Forecast integration** as a separate decision layer

That is a clear *scope expansion* and, if executed well, a defensible thesis contribution.

Second, your title introduces a **transformer-based framework**. The base paper focuses on agentic orchestration and interactive model selection; it does not center the contribution on a specific transformer forecasting backbone, but rather on the interactive workflow and model search/deployment. citeturn9view0turn10view0turn10view3  
So adding a strong transformer backbone (PatchTST/iTransformer) is a modeling-level novelty. citeturn13view0turn13view1  

Third, you can define “Agentic AI” novelty precisely as:
- a **generalizable multi-agent workflow** that can run *two forecasting pipelines* (load + solar) and then fuse outputs,
- with a **formal integration agent** that resolves inconsistencies and produces decision-ready summaries (as your slides say),
- while preserving the base paper’s rigor: topic-based messaging, separated planning/execution, and logged postprocessing. citeturn9view0turn10view3turn16view0  

If you want a thesis-grade novelty statement in one sentence:

> “We extend an LLM-driven interactive forecasting workflow from load-only to a joint load–solar setting by introducing solar-specialized training and an explicit forecast-integration agent, and we evaluate transformer-based forecasting backbones within this agentic pipeline under reproducible, time-series evaluation protocols.” citeturn9view0turn3search3turn13view0