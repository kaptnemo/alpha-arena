# Alpha Arena

`Alpha Arena` 是一个面向 **A 股面板时序建模** 的**严肃量化研究原型**。  
它关注的不是单个模型是否“跑通”，而是把数据口径、特征构造、训练流程、结构对比与研究结论沉淀在同一套工程中，用于回答：

> 在严格避免数据泄漏的前提下，针对 CSI 300 等股票池的历史序列与横截面特征，哪些时序建模结构能够在 **IC / Rank IC / 分组收益** 上稳定优于可信基线？

---

## 1. 项目定位

这个仓库当前的合理定位是：

| 定位 | 判断 |
| --- | --- |
| Demo | 已经超过 |
| 研究原型 | 当前定位 |
| 可交易级 alpha 系统 | 还未达到 |

它已经具备数据、特征、模型、训练的主链路，但还在补齐评估、回测、基线体系与 ablation 框架。  
因此，这个项目**不应被理解为“一个最终策略”**，而应被理解为：

- 一个面向 A 股时序建模的研究工程骨架
- 一个围绕基线、结构对比和可复现实验管理持续演进的研究平台
- 一个用于沉淀研究结论，而不是堆积零散 notebook 的仓库

---

## 2. 研究目标

项目的中期目标不是“做出更复杂的模型”，而是逐步形成下面这些研究能力：

1. 同一实验的**数据、配置、结果**可以被清晰记录与追踪
2. 传统方法、深度基线与增强结构可以放在**同一口径**下观察表现
3. attention、dual-head、memory cache 等设计可以通过 **ablation** 拆开验证
4. 研究判断最终落在 **IC / Rank IC / 分组收益 / long-short**，而不是停留在训练 loss

对应的总体路线见 [`ROAD_MAP.md`](./ROAD_MAP.md)。

---

## 3. 当前状态

### 已具备

- **数据接入**
  - BaoStock / Tushare 数据抓取
  - parquet / 本地文件加载
- **特征工程**
  - 单股票时序特征
  - panel 横截面特征
  - 标签构造
  - LSTM 输入列筛选
- **模型原型**
  - `AEDH-LSTM`：注意力增强双头 LSTM 基线
  - `AMC-LSTM`：带 segment memory cache 的增强原型
- **训练入口**
  - `src/alpha_arena/train/main.py`
  - `src/notebooks/train_aedh_lstm.py`
  - `src/notebooks/train_aedh_lstm.ipynb`

### 仍在补齐

- 统一实验配置与结果落盘协议
- 统一 prediction schema
- IC / Rank IC / 分组收益 / long-short 评估层
- baseline family（线性、树模型、MLP、plain LSTM）
- 系统化 ablation pipeline
- 回测与交易成本敏感性分析

---

## 4. 研究原则

### 时序安全优先

- 标签仅由未来价格构造
- 数据切分按时间推进，不做随机切分
- 序列样本与 panel 样本保持 `date × ts_code` 对齐

### 先做研究闭环，再堆模型复杂度

没有统一评估与复现协议之前，新增复杂模型只会增加不确定性，不会增加研究可信度。

### 所有改进必须打基线

任何结构升级都必须至少与以下对象比较：

- 朴素规则基线
- 非深度学习基线
- 当前稳定深度基线

### 研究结论必须可复核

一次实验最终应能回答：

- 用了哪份数据
- 用了哪组特征
- 用了哪份配置
- 在哪些窗口上有效
- 相比哪些基线更好
- 提升是否具有稳定性

---

## 5. 当前工程结构

| 模块 | 作用 |
| --- | --- |
| `src/alpha_arena/data/` | 数据抓取、读取与本地落盘 |
| `src/alpha_arena/features/` | 特征构造、标签生成、特征筛选 |
| `src/alpha_arena/models/` | 时序模型定义与结构原型 |
| `src/alpha_arena/train/` | 数据集封装、训练器、训练入口 |
| `src/alpha_arena/cli/` | CLI 数据抓取入口 |
| `src/notebooks/` | notebook 友好的训练脚本与 Jupyter notebook |
| `configs/` | 后续实验配置化入口 |
| `ROAD_MAP.md` | 研究原型演进路线图 |

项目关注的主链路是：

```text
raw market data
  -> data ingestion / load
  -> single-stock temporal features
  -> panel cross-sectional features
  -> future return targets
  -> sequence dataset build
  -> model training
  -> prediction / evaluation / backtest
```

---

## 6. 当前模型路线

| 模型 | 文件 | 当前定位 |
| --- | --- | --- |
| `AEDH-LSTM` | `src/alpha_arena/models/aedh_lstm.py` | 稳定、清晰、可解释的深度基线 |
| `AMC-LSTM` | `src/alpha_arena/models/amc_lstm.py` | 面向长依赖建模的增强结构原型 |

### AEDH-LSTM

核心结构：

- input projection
- multi-layer LSTM
- temporal attention
- `context + last_state` 融合
- dual heads：收益头 + 风险头

研究意义：

- 作为当前的主深度基线
- 便于做 attention / dual-head / loss 设计的 ablation

### AMC-LSTM

核心结构：

- 多层 `LSTMCell` 级联
- segment-level memory cache
- 历史片段检索与融合
- temporal attention + dual heads

研究意义：

- 用于检验更强的长依赖建模是否带来稳定提升
- 重点不是“更复杂”，而是验证 memory 机制是否真的有效

---

## 7. 特征与标签设计

项目当前的特征工程围绕两类信息组织：

### 单股票时序特征

- 收益率与价格行为
- 波动率与风险代理
- 成交量与成交活跃度
- `ta` / `pandas-ta` 技术指标
- 时间周期编码

### 横截面特征

- `*_cs_rank`
- `*_cs_z`

这类特征用于补充同一交易日截面内的相对强弱信息。

### 标签构造

当前标签设计遵守以下原则：

- 在同一股票内部按时间 `shift(-h)` 构造未来收益
- 不跨股票错位
- 末尾自然产生缺失标签，训练时过滤

### 输入稳定性处理

项目对数值特征使用 rolling z-score，目标是：

- 缓解金融时间序列的非平稳性
- 控制不同特征量纲差异
- 改善 LSTM 类模型训练稳定性

---

## 8. 为什么它不是“又一个模型 demo”

这个仓库的重点不是单独展示某个模型结构，而是把下面几层边界拆开：

- **data**：数据从哪里来，怎么落盘
- **features**：模型看什么，标签怎么定义
- **models**：序列如何编码
- **train**：如何训练并保存 history
- **eval / backtest**：结果如何判断是否真实有效

真正的目标不是“跑出一个低 loss”，而是让未来能够系统回答：

- attention 是否优于只看最后一步
- dual-head 是否提升排序稳定性
- memory cache 是否真正改善长依赖建模
- 某个结构的提升是否跨年份、跨市场状态存在

---

## 9. 快速开始

### 安装依赖

```bash
poetry install
```

### 数据抓取

```bash
poetry run arena_cli index_stocks 2017 2025 --index-name csi300 --storage-format parquet
poetry run arena_cli stock_daily sh.600000 20200101 20251231 --storage-format parquet
```

### 运行训练脚本

```bash
poetry run python src/notebooks/train_aedh_lstm.py
```

这个脚本会：

- 加载 `SequenceDataset`
- 启动单机训练
- 输出训练日志
- 保存 checkpoint / history
- 绘制并保存 loss 曲线

### 使用 Jupyter Notebook

打开：

```text
src/notebooks/train_aedh_lstm.ipynb
```

Notebook 当前适合：

- 快速检查训练链路
- 观察日志输出
- 查看 loss 曲线

它目前**不是**实验管理与评估的最终形态。

---

## 10. 下一阶段的关键工作

按照当前路线，优先级最高的三件事是：

1. **补齐统一评估与分组回测**
2. **建立传统基线 + 深度基线对照组**
3. **把 AEDH-LSTM / AMC-LSTM 放进统一 ablation 框架**

这三步完成后，项目才会从“研究工程骨架”真正进入“可产出研究结论”的状态。

---

## 11. 当前边界与预期管理

这个项目当前**不宣称**：

- 已经形成可交易级策略
- 已经验证稳定实盘 alpha
- 已经完成严格回测闭环

这个项目当前**可以合理宣称**：

- 已经形成 A 股时序建模的工程主链路
- 已经具备做基线与结构对比研究的基础
- 正在朝“严肃量化研究原型”推进

---

## 12. 参考文档

- [`ROAD_MAP.md`](./ROAD_MAP.md)：项目演进路线图
- `src/alpha_arena/models/`：模型原型
- `src/alpha_arena/train/`：训练主链路
- `src/notebooks/train_aedh_lstm.py`：notebook 友好训练脚本
- `src/notebooks/train_aedh_lstm.ipynb`：Jupyter 训练入口
