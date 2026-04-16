# Alpha Arena

面向 **CSI 300 / A 股面板时序建模** 的研究工程，目标是把“可运行的量化研究代码”与“可扩展的序列模型实验框架”放在同一个仓库里，而不是只做一个单独可跑的模型脚本。

当前仓库已经具备三块核心能力：

1. **数据接入**：支持 BaoStock / Tushare 数据抓取与本地落盘。
2. **特征工程**：支持单股票时序特征、panel 横截面特征、目标构造与 LSTM 输入特征筛选。
3. **模型原型**：提供基线版 `AEDH-LSTM` 与增强版 `AMC-LSTM`，用于对比不同序列建模设计。

---

## 1. 项目目的

这个项目关注的是一个典型的量化研究问题：

> 在严格避免数据泄漏的前提下，基于沪深 300 股票的历史序列特征，学习一个同时具备 **收益预测能力** 与 **风险刻画能力** 的时序表示。

落到工程上，项目不是单纯追求“把 LSTM 跑起来”，而是强调下面几件事：

- **时间序列安全**：只允许按时间先后构造样本、标签和切分数据。
- **面板数据一致性**：样本按 `date × symbol` 组织，避免跨股票、跨时间错误对齐。
- **研究可复现**：特征构造、标签生成、模型结构都拆成模块，方便做 ablation 与迭代。
- **工程可演进**：数据层、特征层、模型层彼此解耦，后续可以继续接训练、评估、回测模块。

---

## 2. 工程化设计重点

这个仓库的工程化重点，不是“堆很多目录”，而是把研究流程拆成清晰边界。

| 模块 | 作用 | 工程价值 |
| --- | --- | --- |
| `src/alpha_arena/data/` | 数据抓取、加载、本地落盘 | 统一数据入口，减少 notebook 式散乱脚本 |
| `src/alpha_arena/features/` | 特征配置、基础特征、技术指标、目标构造、特征筛选 | 保证特征逻辑可复用、可组合、可追踪 |
| `src/alpha_arena/models/` | 时序模型定义 | 支持基线模型与增强模型做结构对比 |
| `src/alpha_arena/cli/` | 命令行入口 | 便于批量抓取数据与固定化流程 |
| `configs/` | 预留实验配置目录 | 便于后续训练、评估、回测配置化 |
| `notebooks/` | 探索性分析 | 与核心逻辑隔离，避免 notebook 反向污染工程代码 |

### 当前代码体现出的工程原则

- **模块拆分明确**：`pipeline.py` 仅做兼容导出，实际逻辑已拆到 `builder / targets / selector / config` 等子模块。
- **避免未来信息泄漏**：标签通过 `groupby(...).shift(-h)` 构造，且仅在同一股票内部平移。
- **单股时序与 panel 横截面分层处理**：先按股票独立做时序特征，再在主进程补横截面 rank / z-score。
- **输入尺度控制**：对数值特征做 rolling z-score，适合 LSTM 类模型训练稳定性。
- **模型对比友好**：同仓库同时保留基线版和增强版，方便做结构 ablation，而不是一次性写死终版模型。

---

## 3. 项目整体流程

```text
原始行情数据
   -> 数据抓取 / parquet 读取
   -> 单股票时序特征构造
   -> panel 横截面特征补充
   -> 未来收益标签构造
   -> LSTM 输入列筛选
   -> 序列模型训练 / 推断
   -> 收益 / 风险输出
```

一个更接近工程视角的理解是：

- **data** 解决“数据从哪里来”
- **features** 解决“模型到底看什么”
- **models** 解决“序列信息怎么编码”
- 后续的 **train / eval / backtest** 解决“结果如何验证”

---

## 4. 模型结构对比

仓库当前包含两条清晰的模型路线：

| 模型 | 文件 | 核心结构 | 适用定位 |
| --- | --- | --- | --- |
| **AEDH-LSTM** | `src/alpha_arena/models/aedh_lstm.py` | Input Projection + LSTM + Temporal Attention + Dual Heads | 作为稳定、清晰的注意力增强基线 |
| **AMC-LSTM** | `src/alpha_arena/models/amc_lstm.py` | Input Projection + Multi-Layer LSTMCell Stack + Segment Memory Cache + Temporal Attention + Dual Heads | 作为面向长序列建模的增强版结构 |

### 4.1 AEDH-LSTM：注意力增强双头 LSTM 基线

`AttentionEnhancedDualHeadLSTM` 的核心思想是：

- 用共享 LSTM 编码器抽取时间序列表示
- 用 **Temporal Attention** 在时间维度上聚焦关键时刻
- 将 `context` 与 `last_state` 融合
- 输出两个 head：
  - `return_head`：预测未来收益
  - `risk_head`：预测对数方差 / 波动不确定性

这个版本的优势是：

- 结构简单，便于解释和复现
- attention 权重天然可视化
- 可以作为后续结构优化的稳定对照组

### 4.2 AMC-LSTM：加入分段记忆缓存的增强结构

`AttentionEnhancedDualHeadalpha_arena`（AMC-LSTM 原型）在基线基础上继续做了两层增强：

1. **把标准 `nn.LSTM` 换成显式的多层 `LSTMCell` 级联**
   - 更方便对每个时间步做自定义控制
   - 为“边走序列边查记忆”提供插入点

2. **引入 Segment Memory Cache**
   - 按 `segment_len` 将序列切成多个局部片段
   - 每个片段沉淀为 `segment memory / segment context`
   - 当前时刻通过相似度检索历史片段表示
   - 再通过 `residual / gated_residual / topk_gated` 等方式融合当前状态与历史记忆

这类设计更适合回答一个核心问题：

> 当有效信息跨越较长时间跨度，且单纯靠最后一步 hidden state 不够时，能否通过“局部摘要 + 检索式记忆”提升序列表示能力？

---

## 5. 模型差异：为什么要做结构对比

从研究角度，保留两个模型不是重复，而是为了让优化路径可证伪。

| 对比维度 | AEDH-LSTM | AMC-LSTM |
| --- | --- | --- |
| 时序编码 | 标准多层 LSTM | 显式多层 LSTMCell 逐步更新 |
| 长依赖处理 | 主要依赖 LSTM 和记忆门控 | 额外依赖分段缓存检索 |
| 可解释性 | attention 权重清晰 | attention + cache 权重双重解释 |
| 复杂度 | 较低 | 较高 |
| 工程风险 | 小，易训练 | 更强，但需要更谨慎调参与验证 |
| 研究价值 | 稳定基线 | 序列建模增强与 ablation 核心对象 |

建议的研究比较方式：

1. **先用 AEDH-LSTM 跑通基线**
2. **再切换 AMC-LSTM 比较收益 / 风险预测质量**
3. **单独 ablate memory cache**，确认提升来自记忆机制，而不是参数量增加

---

## 6. 序列建模优化：这个仓库到底优化了什么

这个项目的“序列建模优化”不是泛泛而谈，而是体现在具体结构与数据处理选择上。

### 6.1 用 Temporal Attention 替代“只看最后一步”

普通 LSTM 常把最后一步 hidden state 当作整个序列摘要，但在金融序列里，关键事件不一定发生在最后几天。

因此这里引入：

- **时间维 attention scoring**
- **attention-weighted context pooling**
- 与最后时刻状态做融合，而不是二选一

这样做的意义是：让模型能够显式关注某些拐点、放量、波动放大或趋势切换时刻。

### 6.2 用 rolling z-score 优化时序输入稳定性

金融特征量纲差异很大，收益率、成交量比率、ATR、RSI、drawdown 混在一起直接输入，训练会非常不稳定。

仓库中的特征构造已经对数值列进行 **rolling z-score**，这相当于在时间维上做局部标准化，优势是：

- 更贴合非平稳金融时间序列
- 比全样本标准化更安全
- 能减轻不同特征量纲差异对 LSTM 梯度的冲击

### 6.3 用 panel 横截面特征增强相对信息

除了单股票时间序列特征，项目还支持在每个交易日截面上计算：

- `*_cs_rank`
- `*_cs_z`

这有助于把“市场整体涨跌”与“个股相对强弱”区分开。虽然这些特征默认不直接输入 LSTM，但它们为后续多任务学习、横截面排序和因子评估留出了空间。

### 6.4 用 Segment Memory Cache 增强长序列依赖

AMC-LSTM 的关键优化在于：

- 把长序列拆成多个 segment
- 为 segment 构造摘要记忆
- 当前时刻检索最相关的历史片段
- 再做 gated fusion

相比把所有信息都压进单个 hidden state，这种方式更像一种 **压缩式长期记忆机制**，适合处理：

- 市场 regime 切换
- 相似波动形态的重复出现
- 需要跨窗口回溯的模式匹配

### 6.5 用双头输出同时建模收益与风险

当前双头设计不是“两个任务凑一起”，而是服务于量化研究中的两个关键输出：

- **收益头**：预测未来回报方向与幅度
- **风险头**：预测 log-variance / volatility proxy

这让模型不仅能给出“看多还是看空”，还能给出“这个判断有多不确定”，对后续风险调整、排序和组合构建都更友好。

---

## 7. 特征工程设计

特征工程部分体现了较强的工程抽象，核心包括：

- **基础收益率与价格行为特征**
- **波动率 / 成交量 / 风险调整特征**
- **`ta` 与 `pandas-ta` 技术指标**
- **时间周期编码**
- **panel 横截面排名与标准化**
- **未来收益标签构造**

其中几个关键点非常重要：

### 标签构造

`src/alpha_arena/features/targets.py`

- 使用 `groupby("symbol").shift(-h)` 构造 `y_ret_h`
- 标签只依赖未来价格
- 不跨股票错位
- 序列末尾自然产生 NaN，需要在训练切片时去掉

### 特征筛选

`src/alpha_arena/features/selector.py`

- 自动排除原始 OHLCV 列
- 默认排除横截面特征列
- 仅保留符合命名规则的数值特征

这使得“原始数据 -> 特征表 -> LSTM 输入张量”之间的转换更加稳定，不容易因为手工选列失误引入噪声。

---

## 8. 目录结构

```text
alpha-arena/
├── configs/
├── data/
│   └── raw/
├── docs/
├── logs/
├── notebooks/
├── src/
│   └── alpha_arena/
│       ├── cli/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── train/
│       └── utils/
├── pyproject.toml
└── README.md
```

---

## 9. 快速开始

### 安装依赖

```bash
poetry install
```

### 使用 CLI 抓取数据

```bash
poetry run arena_cli csi300-stocks 2017 2025 --storage-format parquet
poetry run arena_cli stock-daily sh.600000 20200101 20251231 --storage-format parquet
```

### 在代码里使用特征工程与模型

```python
from alpha_arena.features.builder import build_panel_features
from alpha_arena.features.targets import add_targets
from alpha_arena.features.selector import select_lstm_feature_columns
from alpha_arena.models.aedh_lstm import AttentionEnhancedDualHeadLSTM, AEDH_LSTMConfig

panel_df = build_panel_features(raw_df)
panel_df = add_targets(panel_df, horizons=(1, 5, 10))
feature_cols = select_lstm_feature_columns(panel_df)

model = AttentionEnhancedDualHeadLSTM(
    AEDH_LSTMConfig(input_dim=len(feature_cols))
)
```

---

## 10. 研究视角下的推荐实验路径

如果把这个仓库当成一个量化研究工程，比较合理的实验顺序是：

1. **先固定数据集与特征口径**
2. **用 AEDH-LSTM 建立 attention 基线**
3. **再切换到 AMC-LSTM 检验记忆缓存是否提升长序列建模**
4. **对比收益预测、风险预测、IC / Rank IC 与分组收益表现**

核心不是追求一次性最复杂模型，而是让每次结构升级都能回答一个明确问题：

- attention 是否真的比纯 last hidden state 更有效？
- memory cache 是否真的提升了长依赖建模？
- 风险头是否提高了预测稳定性与可用性？

---

## 11. 当前状态与后续扩展

当前仓库已经完成了：

- 数据抓取入口
- 特征工程主链路
- 双模型结构原型

后续最自然的扩展方向是：

- 完整训练器与实验配置
- 评估指标（MAE / RMSE / Accuracy / F1 / AUC / IC / Rank IC）
- quantile grouping 与 long-short backtest
- 系统化 ablation pipeline

---

## 12. 结论

`Alpha Arena` 更像一个 **面向量化时序研究的工程骨架**：

- 不是只放一个模型文件
- 不是只放一个 notebook
- 而是把 **数据、特征、模型、实验对比** 放进一个可以持续演进的结构中

如果用一句话概括这个仓库，它的核心价值是：

> 以工程化方式研究 CSI 300 股票序列表示学习，并围绕 **注意力增强**、**双头预测** 与 **记忆缓存** 持续优化时序建模能力。
