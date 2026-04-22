# Alpha Arena Road Map

## 目标定位

把 `Alpha Arena` 从“可运行的量化模型工程”推进为一个**严肃量化研究原型**，形成一套：

1. **可复现** 的 A 股序列建模研究框架
2. **可比较** 的基线 / 增强模型实验体系
3. **可证伪** 的 ablation 流程
4. **可解释** 的评估与回测闭环

最终目标不是“再做一个更复杂的 LSTM”，而是证明：

- 某些结构设计在 **IC / Rank IC / 分组收益 / long-short spread** 上稳定优于基线
- 这种提升不是由数据泄漏、样本偶然性或参数量膨胀造成的
- 研究结论可以被重复运行、重复验证、重复比较

---

## 一句话北极星

> 在严格时序安全和可复现实验管理前提下，建立一条从 A 股面板数据、特征构造、序列建模、评估分析到分组回测的完整研究链路，并用这条链路产出可信的结构比较结论。

---

## 当前定位

项目当前已经具备：

- 数据抓取与落盘
- panel 特征工程与标签构造
- AEDH-LSTM / AMC-LSTM 模型原型
- 基础训练脚本与 notebook 入口

项目当前仍然缺少：

- 统一实验配置与运行协议
- 面向研究结论的评估层
- 明确的 baseline family
- 系统化 ablation pipeline
- 分组收益 / long-short / turnover / cost 分析
- “一次实验对应一份完整产物”的结果管理规范

因此，当前最合理的定位是：

| 阶段 | 判断 |
| --- | --- |
| Demo | 已经超过 |
| 研究原型 | 正在形成 |
| 严肃量化研究框架 | 取决于后续评估与复现体系能否补齐 |

---

## 总体原则

### 1. 先做研究闭环，再堆模型复杂度

如果没有统一评估和可复现协议，新增复杂模型只会增加叙事，不会增加研究可信度。

### 2. 所有改进都必须打基线

任何“提升”都必须至少对比：

- 朴素基线
- 非深度学习基线
- 当前最稳定深度基线

### 3. 所有结果都必须可落盘、可追踪、可重跑

一次实验至少应产出：

- 配置快照
- 数据版本标识
- 训练 history
- 评估指标
- 分组收益结果
- 关键图表
- 简短结论

### 4. 先证明“有效”，再讨论“可交易”

研究阶段优先回答：

- 模型是否真的提取到超越基线的预测信息？
- 信息是否跨年份、跨市场状态稳定存在？

在此之后，再进入成本、容量、组合构建和实盘约束。

---

## 路线图总览

| Phase | 主题 | 核心目标 |
| --- | --- | --- |
| Phase 0 | 研究底座清理 | 先把实验协议、目录、产物定义固定下来 |
| Phase 1 | 数据与样本口径固化 | 确保数据集、切分、标签、特征完全可复现 |
| Phase 2 | 评估闭环建立 | 补齐 IC / Rank IC / 分组收益 / long-short 分析 |
| Phase 3 | 基线体系建立 | 建立传统模型与深度模型的标准对照组 |
| Phase 4 | Ablation 框架建立 | 让 attention、dual-head、memory cache 可证伪 |
| Phase 5 | 增强模型验证 | 在统一协议下验证 AMC-LSTM 等增强结构 |
| Phase 6 | 研究报告化 | 固化结论、图表和实验记录，形成可持续研究框架 |

---

## Phase 0：研究底座清理

### 目标

把项目从“能跑”推进到“知道每次运行到底在做什么”。

### 关键交付物

1. **统一实验目录规范**
   - `configs/experiments/`
   - `outputs/experiments/<run_id>/`
   - `reports/`
   - `src/alpha_arena/eval/`
   - `src/alpha_arena/backtest/`

2. **统一 run artifact 定义**
   每次实验固定输出：
   - `config.json`
   - `dataset_meta.json`
   - `history.json`
   - `metrics.json`
   - `predictions.parquet`
   - `group_returns.parquet`
   - `summary.md`
   - `plots/`

3. **统一实验命名协议**
   推荐包含：
   - model name
   - dataset version
   - label horizon
   - feature set
   - seed

4. **统一随机种子与运行环境记录**
   - seed
   - torch / python / package versions
   - git commit hash

### 完成标准

- 同一实验可以被重新执行
- 重新执行后的核心指标误差在可接受范围内
- 不再依赖 notebook 手工记录实验结果

---

## Phase 1：数据与样本口径固化

### 目标

确保所有模型都在同一份可追踪的数据口径上比较。

### 关键任务

1. **定义 canonical dataset spec**
   - 股票池：CSI 300
   - 时间区间
   - 标签 horizon
   - 序列长度
   - step size
   - 特征列集合
   - 横截面特征是否进入模型

2. **明确 split protocol**
   - train / valid / test 按时间切分
   - 禁止随机切分
   - 预留 walk-forward / rolling evaluation 方案

3. **落盘 dataset version**
   - dataset build config
   - feature column list
   - target column definition
   - row counts by split
   - date range by split

4. **增加数据健康检查**
   - label 缺失率
   - 特征缺失率
   - 样本量按年份/月份分布
   - 每日股票覆盖数
   - 是否存在异常重复样本

### 完成标准

- 任意实验都能明确回答“用了哪一版数据”
- 训练、评估、回测之间不再出现样本口径漂移
- 不同模型比较时只变模型，不变数据定义

---

## Phase 2：评估闭环建立

### 目标

从“loss 驱动”切换到“研究指标驱动”。

### 必须实现的指标

#### 横截面预测指标

- IC
- Rank IC
- IC mean / std
- ICIR
- 按年份 IC / Rank IC
- 按市场状态分层 IC

#### 分组收益指标

- quantile group return
- top-bottom spread
- long-short cumulative return
- group monotonicity

#### 可交易性相关指标

- turnover
- holding period sensitivity
- transaction cost sensitivity
- hit ratio

#### 风险侧指标

- 行业暴露
- 市值风格暴露
- 波动暴露
- 最大回撤
- 夏普 / 信息比率

### 关键任务

1. 新增 `src/alpha_arena/eval/metrics.py`
2. 新增 `src/alpha_arena/eval/panel_eval.py`
3. 新增 `src/alpha_arena/backtest/quantile.py`
4. 统一 prediction schema
   - `date`
   - `ts_code`
   - `y_true`
   - `y_pred`
   - `pred_risk`
   - `split`

5. 输出统一图表
   - IC time series
   - cumulative long-short
   - quantile curves
   - prediction histogram
   - calibration / residual plots

### 完成标准

- 一次实验结束后自动生成研究可读的指标与图表
- “某模型比另一个模型更好”不再只基于 valid loss

---

## Phase 3：基线体系建立

### 目标

建立可信对照组，避免深度模型自说自话。

### 必须包含的基线

#### 朴素基线

- 下一期收益均值
- 过去动量 / 反转简单规则
- 最后一步特征线性映射

#### 传统机器学习基线

- Linear / Ridge / Lasso
- LightGBM / XGBoost
- Random Forest（可选）

#### 深度学习基线

- MLP
- Plain LSTM
- LSTM + last hidden state
- AEDH-LSTM

### 核心要求

1. 所有基线使用同一份训练集 / 验证集 / 测试集
2. 所有基线输出统一 prediction schema
3. 所有基线进入同一评估框架

### 完成标准

- 项目可以回答“深度模型是否真的打败简单方法”
- AEDH-LSTM 的定位从“默认主角”变成“经比较后的稳定深度基线”

---

## Phase 4：Ablation 框架建立

### 目标

让模型设计从“感觉合理”变成“有证据支持”。

### AEDH-LSTM 必做 ablation

1. `last_state only` vs `attention pooling`
2. `attention` vs `attention + last_state`
3. `return head only` vs `dual head`
4. `gaussian_nll` vs `mse`
5. `with cs features` vs `without cs features`
6. `with rolling z-score` vs `without rolling z-score`

### AMC-LSTM 必做 ablation

1. `AEDH-LSTM` vs `AMC-LSTM`
2. `memory on` vs `memory off`
3. `different segment_len`
4. `different memory fusion`
5. `same parameter budget` 下比较

### 关键要求

- 所有 ablation 固定数据口径
- 控制参数量与训练预算
- 固定随机种子组
- 至少比较多次重复运行结果

### 完成标准

- 每一项结构设计都能回答“为什么保留它”
- 结论来自指标与分组收益，而不是单次跑出来的最好结果

---

## Phase 5：增强模型验证

### 目标

只在基线和评估体系稳定后，推进更强模型。

### 候选方向

1. **AMC-LSTM 完整验证**
   - 长依赖是否提升 Rank IC
   - 是否改善 regime shift 下稳定性

2. **多任务 / 多 horizon 学习**
   - 同时预测 `y_ret_5 / y_ret_10 / y_ret_20`
   - 观察是否提升共享表示稳定性

3. **风险建模增强**
   - 风险头是否改善排序稳定性
   - 用风险输出做风险调整排序

4. **panel-aware 建模**
   - 序列表示 + 横截面打分解耦
   - 探索 sequence encoder + cross-sectional head

### 进入本阶段的前提

- 已经有稳定基线
- 已经有统一评估
- 已经有 ablation 协议

### 完成标准

- 增强模型在测试窗口和重复实验中优于基线
- 提升具有统计稳定性，而非偶然峰值

---

## Phase 6：研究报告化与成果固化

### 目标

把实验结果沉淀成可以复用和持续扩展的研究资产。

### 关键交付物

1. **标准实验报告模板**
   - 问题定义
   - 数据口径
   - 模型配置
   - 评估结果
   - 图表
   - 结论
   - 失败案例

2. **baseline leaderboard**
   - 按数据集版本、标签 horizon、测试区间组织

3. **ablation summary**
   - 哪些设计有效
   - 哪些设计无效
   - 哪些设计只在特定窗口有效

4. **研究结论库**
   推荐按“可复用结论”记录，例如：
   - rolling z-score 对 LSTM 稳定训练是否必要
   - dual-head 对排序指标是否有帮助
   - memory cache 是否只在长序列下有效

### 完成标准

- 新实验可以复用老框架，而不是重新搭脚手架
- 项目可以持续积累研究结论，而不是只积累零散脚本

---

## 推荐的实验矩阵

### 数据维度

- 股票池：CSI 300
- horizon：5 / 10 / 20
- sequence length：20 / 60 / 120
- feature set：基础时序 / 时序 + 横截面 / 精简特征

### 模型维度

- Linear
- LightGBM
- MLP
- Plain LSTM
- AEDH-LSTM
- AMC-LSTM

### 重复维度

- 多 seed
- 多测试区间
- 多市场状态

### 输出维度

- IC / Rank IC
- 分组收益
- long-short spread
- turnover
- 风险暴露

---

## 项目成功标准

当以下条件满足时，项目可以被认为已经从“工程原型”升级为“严肃量化研究原型”：

1. **复现性**
   - 同一实验可以稳定重跑
   - 所有关键产物自动保存

2. **可比较性**
   - 至少有一组传统基线、一组深度基线和一组增强模型

3. **可证伪性**
   - attention、dual-head、memory cache 都有明确 ablation

4. **研究有效性**
   - 至少一个增强结构在 IC / Rank IC / 分组收益上稳定优于基线

5. **结论沉淀**
   - 结果不是停留在 notebook，而是形成报告、图表和配置快照

---

## 不应过早投入的方向

在以下基础未完成前，不建议优先投入：

- 继续增加更复杂模型结构
- 大规模调参搜索
- 实盘接口开发
- 多市场扩展
- 花哨的可视化面板

原因很简单：没有研究闭环之前，复杂化只会放大不确定性。

---

## 下一步优先级

如果只做三件事，优先顺序应是：

1. **补齐统一评估与分组回测**
2. **建立传统基线 + 深度基线对照组**
3. **把 AEDH-LSTM / AMC-LSTM 放进统一 ablation 框架**

这三步完成后，项目就不再只是“有意思的模型工程”，而会开始具备真正的研究产出能力。
