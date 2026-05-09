# Alpha Arena

> A research-first AI project for A-share panel modeling.

`Alpha Arena` 是一个面向 **A 股量化研究** 的 AI 项目原型，目标不是单独展示某个模型，而是搭建一条从 **数据接入 -> 特征工程 -> 序列建模 -> 评估分析** 的完整研究链路，用来验证模型是否真的在 **IC / Rank IC / 分组收益** 上带来稳定增益。

它更像一个 **financial time-series research stack**，而不是一个“已经可实盘”的交易系统。

## Why Alpha Arena

大多数时序项目只回答“模型能不能跑”，这个仓库想回答的是：

- 数据口径是否时序安全
- 特征和标签是否可复现
- 模型提升是否能在统一评估下成立
- attention / dual-head / memory cache 是否真的有效

换句话说，`Alpha Arena` 关注的不只是模型结构，而是 **研究闭环本身**。

## Highlights

- **Research-first**：先做评估闭环，再堆模型复杂度
- **Time-series safe**：强调避免 look-ahead bias 和数据泄漏
- **Panel-aware**：面向 `date x ts_code` 的 A 股面板建模场景
- **Modular pipeline**：数据、特征、模型、训练、评估分层清晰
- **Built for ablations**：适合做结构对比、标签实验和基线对照

## Current Scope

当前仓库已经覆盖以下主链路：

1. **Data ingestion**
   基于 Tushare 抓取股票日线和指数成分数据，支持 `CSI 300 / CSI 500 / CSI 1000 / SSE 50` 等研究场景。

2. **Feature engineering**
   提供基础价格特征、收益率特征、波动率特征、风险调整特征、技术指标、时间编码，以及横截面 rank / z-score 特征。

3. **Dataset building**
   将 panel 数据转换为 LSTM 可用的序列样本，按年份切分 `train / evaluate / test`，并结合交易日历构造时序安全的样本锚点。

4. **Modeling**
   当前包含两条主要模型线：
   - `AEDH-LSTM`：Attention-Enhanced Dual-Head LSTM
   - `AMC-LSTM`：带 segment memory cache 的增强原型

5. **Evaluation**
   已具备按日期计算 `IC / Rank IC`、按预测值分组、long-short spread 汇总，以及研究报告型文本摘要。

## Project Structure

```text
src/alpha_arena/
├── cli/          # Typer CLI
├── data/         # 数据抓取、加载、外部数据源辅助
├── evaluation/   # IC、分组、报告分析
├── features/     # 特征工程、标签生成、特征筛选
├── models/       # AEDH-LSTM / AMC-LSTM
├── train/        # 数据集、采样器、训练器、训练入口
└── utils/        # 日志与通用工具

data/
├── raw/
├── processed/
└── dataset/

checkpoints/      # 训练产物
evaluations/      # 评估产物
docs/             # 设计文档与模型笔记
```

## Models

### AEDH-LSTM

当前主深度基线，核心结构包括：

- input projection
- stacked LSTM
- temporal attention
- `context + last_state` fusion
- dual heads: return head + risk head

适合做 attention、loss、cross-sectional feature 等方向的 ablation。

### AMC-LSTM

面向更长依赖建模的增强结构，核心思想是：

- 将序列切成 segment
- 缓存历史 segment 表征
- 在当前时刻做记忆检索与融合

重点不是“更复杂”，而是验证 **memory mechanism** 是否真的在研究指标上带来稳定收益。

## End-to-End Workflow

```text
Tushare / market data
  -> raw panel data
  -> feature engineering
  -> future return targets
  -> sequence dataset build
  -> model training
  -> prediction
  -> IC / Rank IC / grouping analysis
```

## Quickstart

### 1. Install

项目使用 Poetry，Python 版本要求见 `pyproject.toml`。

```bash
poetry install
```

### 2. Configure data source

项目会从环境变量读取 `TUSHARE_TOKEN`：

```bash
export TUSHARE_TOKEN="your_token"
```

### 3. Explore the CLI

```bash
poetry run arena_cli --help
```

当前 CLI 提供两个入口：

- `stock-daily`
- `index-stocks`

### 4. Fetch market data

抓取单只股票日线：

```bash
poetry run arena_cli stock-daily sh.600000 20200101 20201231
```

抓取指数成分股区间数据：

```bash
poetry run arena_cli index-stocks 2019 2024 --index-name csi300
```

## Evaluation Mindset

这个项目不把 `loss` 作为最终结论，而更关注：

- `IC`
- `Rank IC`
- prediction grouping
- long-short spread
- 稳定性和可重复性

所以它更接近一个 **AI for quantitative research** 项目，而不是单纯的深度学习 demo。

## Current Status

这个仓库当前最适合：

- 做 A 股 panel 时序建模实验
- 验证特征、标签和模型结构
- 做 baseline 对比和 ablation
- 沉淀统一的数据与评估口径

当前仍在继续完善：

- 统一实验配置协议
- 更标准化的训练/评估入口
- baseline family
- 更完整的回测与报告产物

## Reading Order

如果你第一次进入这个仓库，推荐这样阅读：

1. `ROAD_MAP.md`：先理解项目的研究目标
2. `src/alpha_arena/features/`：理解特征和标签口径
3. `src/alpha_arena/train/dataset/builder.py`：理解时序样本构造
4. `src/alpha_arena/models/`：理解 AEDH-LSTM / AMC-LSTM
5. `src/alpha_arena/evaluation/`：理解“如何判断模型有效”

## Testing

仓库当前带有 `src/tests/` 下的单元测试，覆盖重点包括：

- 数据接入配置
- 特征构建一致性
- 数据集切分与采样策略
- 分组评估
- 报告汇总逻辑

```bash
pytest -q
```

## Roadmap

按照现有路线，接下来的重点是：

1. 补齐统一评估与分组回测
2. 建立传统基线与深度基线对照组
3. 将 AEDH-LSTM / AMC-LSTM 纳入统一 ablation 框架

更多背景见 [`ROAD_MAP.md`](./ROAD_MAP.md)。

## Disclaimer

这是一个 **research prototype**。
它用于研究和验证 alpha 建模思路，不应被直接视为可实盘部署的交易系统或收益承诺。
