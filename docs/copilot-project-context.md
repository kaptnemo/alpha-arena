# Copilot Instructions for This Repository

## Project Overview
This repository implements an **Attention-Enhanced Dual-Head LSTM with Memory Caching** for **CSI 300 (沪深300) stock modeling and analysis**.

The model is designed for **multi-task time-series learning** on A-share equities, where:

- The **shared encoder** captures temporal market dynamics from sequential stock features
- The **attention mechanism** emphasizes informative historical timesteps
- The **dual heads** support two target tasks:
  - **Head 1**: return prediction / alpha regression
  - **Head 2**: trend direction / up-down classification
- The **memory caching module** stores useful historical latent patterns or market states to improve stability and adaptivity

The repo should support:
- data preprocessing for CSI 300 constituent stocks
- rolling-window training / validation / testing
- factor-style feature engineering
- model training, evaluation, backtesting, and ablation studies
- reproducible experiments and clean modular code

---

## Core Development Principles

### 1. Prioritize correctness over cleverness
When generating code:
- prefer readable, explicit implementations
- avoid hidden side effects
- avoid overly compact one-liners in core logic
- do not introduce magic constants without explanation
- document tensor shapes in comments where useful

### 2. Time-series safety is mandatory
All generated code must avoid look-ahead bias and data leakage.
Always assume:
- labels must be generated strictly from future returns relative to feature timestamps
- training / validation / testing splits must be chronological
- normalization / standardization must be fit on training data only
- rolling windows must not leak future information

If there is any ambiguity, choose the implementation that is safest for financial time-series research.

### 3. Modular architecture
Prefer the following code organization:
- `data/` for raw and processed datasets
- `configs/` for YAML or JSON experiment configs
- `src/data/` for loaders, preprocessors, feature builders
- `src/models/` for LSTM, attention, dual heads, memory cache
- `src/train/` for training loop, loss, metrics, scheduler
- `src/eval/` for evaluation, IC, rank IC, classification metrics
- `src/backtest/` for simple strategy simulation and portfolio analysis
- `src/utils/` for logging, seed control, config, plotting
- `scripts/` for entry scripts
- `notebooks/` only for exploration, never core production logic

When writing new code, place it in the most appropriate module rather than making monolithic files.

### 4. Reproducibility first
Always:
- set random seeds
- make device selection explicit
- log config values
- save checkpoints and metrics
- separate config from code where possible

### 5. Production-quality research code
Code should be suitable for iterative quantitative research:
- typed function signatures when practical
- docstrings for public functions/classes
- robust error handling for data issues
- clear logging messages
- minimal but useful unit-testable components

---

## Domain Context

### Market scope
- Universe: CSI 300 constituent stocks, or stock pool aligned to CSI 300 research setting
- Frequency: default to daily bars unless config specifies otherwise
- Typical inputs may include:
  - OHLCV-derived features
  - return-based features
  - volatility and momentum indicators
  - turnover / amount / volume features
  - technical indicators
  - cross-sectional rank features
  - industry / sector encodings if available
  - benchmark-relative features if available

### Typical targets
Support configurable target definitions, such as:
- next-day return
- next 5-day return
- future excess return over benchmark
- binary direction label
- ternary movement label

Default assumptions:
- regression target is future return
- classification target is future direction based on thresholded return

### Evaluation priorities
When implementing evaluation, prefer finance-relevant metrics:
- IC (Information Coefficient)
- Rank IC
- RMSE / MAE for regression
- Accuracy / F1 / AUC for classification
- precision@k or hit rate@k when useful
- long-short spread analysis
- grouped return analysis by prediction quantiles
- cumulative return and max drawdown for simple backtest views

---

## Model Design Expectations

### Shared encoder
The shared encoder should usually include:
- input projection if needed
- LSTM stack (`batch_first=True`)
- optional dropout between layers
- optional layer normalization

### Attention module
The attention mechanism should:
- operate on sequence hidden states
- output attention weights over timesteps
- produce a context vector for downstream heads
- expose attention weights for visualization / interpretability

When writing attention code:
- keep tensor shapes explicit
- support masking if padded sequences are used
- return both context and attention weights when helpful

### Dual-head outputs
The model should have two task heads:
- a regression head for return prediction
- a classification head for direction prediction

Expected behavior:
- shared temporal representation
- separate head-specific MLP or linear layers
- configurable loss weights for multi-task training

### Memory caching
The memory caching module should be implemented as a clearly isolated component.

Possible responsibilities:
- maintain a cache of recent latent states, prototypes, or market regimes
- retrieve similar past representations
- fuse retrieved memory with current context
- support configurable cache size and update policy

When generating memory cache code:
- keep the interface simple and explicit
- document cache update timing
- prevent accidental gradient misuse if cache stores detached states
- make it easy to ablate by disabling the cache via config

### Loss design
Default multi-task loss:
- regression loss: MSE or Huber
- classification loss: BCEWithLogitsLoss or CrossEntropyLoss
- total loss = `lambda_reg * loss_reg + lambda_cls * loss_cls`

Make the loss configurable from experiment config.

---

## Coding Preferences

### Python style
- Target Python 3.10+
- Use `pathlib` instead of raw string paths
- Prefer `dataclass` or pydantic-like config patterns when useful
- Use `logging` instead of scattered `print`
- Keep functions short and focused
- Avoid deeply nested logic when a helper function would improve clarity

### PyTorch style
- Use `nn.Module` cleanly
- keep forward passes readable
- annotate important tensor shapes in comments
- avoid unnecessary in-place operations
- handle device transfer explicitly
- write train/eval loops that clearly separate:
  - forward
  - loss computation
  - backward
  - optimizer step
  - metric logging

### Pandas / data handling style
- write preprocessing steps as deterministic functions
- avoid chained assignment
- validate sorted timestamps and stock IDs
- ensure panel data indexing is explicit (`date`, `symbol`)
- prefer parquet / feather / pickle only when justified; csv is acceptable for portability

---

## Financial Research Constraints

### Always guard against these mistakes
- look-ahead bias
- survivorship bias where avoidable
- leakage from future constituent changes
- fitting scalers on all data
- shuffling time-series samples across time splits
- mixing stock identities incorrectly across windows
- using future returns or future-adjusted fields as current features

### Backtest simplification rules
If asked to implement a backtest:
- keep it simple, transparent, and clearly labeled as research backtest
- include transaction cost assumptions as configurable parameters
- avoid unrealistic execution assumptions
- separate signal generation from portfolio construction
- do not present toy backtests as production trading systems

---

## Preferred File Patterns

### Good examples
- `src/models/attention.py`
- `src/models/dual_head_lstm.py`
- `src/models/memory_cache.py`
- `src/train/trainer.py`
- `src/eval/metrics.py`
- `src/data/feature_engineering.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `configs/base.yaml`

### Avoid
- massive all-in-one notebooks
- a single `model.py` containing every class
- hidden preprocessing inside training scripts
- duplicated metric logic across files

---

## When Generating Code, Copilot Should

### For new model code
- create small, composable modules
- include docstrings
- expose key hyperparameters in constructors
- include shape-safe forward methods
- make attention and memory modules optional by config

### For new training code
- use chronological splits
- support early stopping
- save best checkpoint by validation metric
- log per-task loss and total loss
- support GPU if available

### For new evaluation code
- report both ML metrics and quant-style metrics
- support cross-sectional daily evaluation
- provide grouped analysis by prediction rank/quantile
- keep outputs easy to export to csv

### For refactors
- preserve behavior unless explicitly asked to change it
- do not silently rename public APIs
- prefer incremental refactoring over sweeping rewrites

### For bug fixes
- identify root cause first
- propose minimal safe patch
- mention any likely downstream impacts
- add assertions or tests when appropriate

---

## Preferred Response Behavior from Copilot

When answering implementation questions in this repository, Copilot should:
1. briefly state assumptions
2. generate code that fits the existing project structure
3. explain key tensor shapes for model code
4. flag any risk of leakage or time-series misuse
5. suggest a minimal validation check or test where appropriate

When requirements are ambiguous, default to:
- daily-frequency stock prediction
- PyTorch implementation
- chronological rolling split
- regression + binary classification dual-task learning
- detachable memory cache for stability

---

## Example Defaults

Unless otherwise specified, use these defaults:
- framework: PyTorch
- sequence length: configurable, default 20 or 30
- optimizer: Adam
- learning rate: configurable, default `1e-3`
- regression loss: HuberLoss
- classification loss: BCEWithLogitsLoss
- batch layout: `[batch, seq_len, feature_dim]`
- output:
  - regression head: `[batch, 1]`
  - classification head: `[batch, 1]`
- attention output:
  - context: `[batch, hidden_dim]`
  - attention weights: `[batch, seq_len]`

---

## Testing Expectations

For important code, generate lightweight tests where possible:
- tensor shape checks
- no-leakage checks in dataset slicing
- cache size / update behavior checks
- forward-pass smoke tests
- train step smoke tests on synthetic data

---

## Documentation Expectations

When generating nontrivial code:
- add concise docstrings
- explain expected input columns for data functions
- document target construction clearly
- document whether tensors are detached before entering memory cache
- note assumptions for backtest calculations

---

## Non-Goals
Unless explicitly requested, do not:
- add web frameworks or UI layers
- add heavy distributed training infrastructure
- over-engineer config systems
- optimize prematurely
- claim real trading profitability from offline experiments

---

下面是已经帮你整理好的 **可直接追加到 `docs/copilot-project-context.md` 的标准评估 pipeline 段落**（已按 Copilot 易理解结构写好，直接粘即可）：

---

## Standard Evaluation Pipeline (Financial + ML)

This project evaluates the model from both **machine learning** and **quantitative finance** perspectives.

The model output (regression + classification) is treated as an **alpha signal / factor**, and must be evaluated using **cross-sectional financial tests**.

---

### 1. Information Coefficient (IC)

Primary evaluation metric for cross-sectional prediction:

[
IC = corr(\hat{r}*t, r*{t+1})
]

* computed **cross-sectionally per date**
* report:

  * mean IC
  * IC standard deviation
  * IC Information Ratio (ICIR)

Also compute:

* **Rank IC** (Spearman correlation)

Purpose:

* measures predictive power of the signal

---

### 2. Quantile Grouping Test

For each date:

* rank stocks by prediction
* split into quantiles (e.g., 5 or 10 groups)

Evaluate:

* average return per group
* monotonicity across groups

Key outputs:

* top group return
* bottom group return
* **long-short spread (top - bottom)**

Purpose:

* tests economic usefulness of predictions

---

### 3. Long-Short Portfolio Backtest (Research Only)

Construct a simple strategy:

* long: top quantile
* short: bottom quantile
* equal-weight or value-weight

Evaluate:

* cumulative return
* Sharpe ratio
* max drawdown
* turnover (if available)

Important:

* clearly label as **research backtest**
* include transaction cost assumptions if possible

---

### 4. Fama-MacBeth Regression

Cross-sectional regression per time step:

[
r_{i,t+1} = \alpha_t + \beta_t \cdot \hat{y}*{i,t} + \epsilon*{i,t}
]

Then:

* average (\beta_t) over time
* compute t-statistics

Purpose:

* test whether model signal has **pricing power**

---

### 5. Multi-Factor Regression (Alpha Test)

Test whether the model provides information beyond known factors:

[
r = \alpha + \beta_1 MKT + \beta_2 SMB + \beta_3 HML + \gamma \cdot \hat{y} + \epsilon
]

Check:

* significance of (\gamma)
* whether (\alpha) remains significant

Purpose:

* test **independent alpha contribution**

---

### 6. Machine Learning Metrics

Also report standard ML metrics:

Regression head:

* MAE
* RMSE

Classification head:

* Accuracy
* F1-score
* AUC

Note:

* these are secondary to financial metrics

---

### 7. Stability & Robustness Tests

Evaluate performance consistency:

* IC over time (rolling IC)
* IC standard deviation
* performance across:

  * bull / bear markets
  * high / low volatility regimes

Purpose:

* test robustness and generalization

---

### 8. Ablation Study (Critical)

Compare:

* LSTM only
* LSTM + Attention
* LSTM + Attention + Memory Caching

Evaluate differences in:

* IC / Rank IC
* Sharpe ratio
* long-short return

Purpose:

* isolate contribution of:

  * attention mechanism
  * memory caching module

---

### 9. Key Principles

* all evaluations must be **time-series safe**
* no future information leakage
* all metrics computed **out-of-sample**
* cross-sectional evaluation must align on the same date

---

### Summary

The model should be evaluated as a **factor generation system**, not just a prediction model.

Primary focus:

* IC / Rank IC
* long-short performance
* statistical significance (Fama-MacBeth)

Secondary focus:

* ML metrics
* training loss

---

如果你下一步想做实战，我可以直接帮你把这一整套 pipeline 变成：

👉 `src/eval/metrics.py` + `backtest.py` 可运行代码版本（直接接你模型输出用）


---

## Summary
This is a **financial time-series research repository** focused on an **Attention-Enhanced Dual-Head LSTM with Memory Caching** for **CSI 300 stock modeling**.

The highest priorities are:
- no leakage
- modular PyTorch code
- reproducible experiments
- interpretable attention
- configurable memory caching
- finance-relevant evaluation