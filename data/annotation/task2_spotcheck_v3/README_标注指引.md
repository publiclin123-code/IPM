# 信号抽查确认（v3 扩容）标注指引 — 预警语义验证

> 版本 v3.0 · 标注对象：`task2_spotcheck_v3/forward_spotcheck_v3.csv`（120 行）
> 对应论文 §4.4 "Models and evaluation design" 的 **forward-signal spot-check（扩容版）**。
> 回答一个窄问题："被 protocol 判定为 forward 的信号，语义上是不是真的'尚未发生的供应链中断预警'"。

---

## 一、当前进度

| 批次 | 行数 | 状态 |
|---|---|---|
| round1（`from_round1=1`） | 52 | **已仲裁完成，不要改**（`human_label` 已填） |
| 新增（`from_round1=0`） | **68** | **待你确认**：`human_label` 为空，`llm_suggest` 已由 LLM 预标注 |

## 二、你要做的只有一件事

对 **`from_round1=0` 的 68 行**，填 **`human_label`** 列：

| 值 | 含义 | 判定 |
|----|------|------|
| `1` | 真预警 | 报道发出时，它所警告的供应链中断（trigger）确实还没发生 |
| `0` | 非预警 | 不是预警：报道的是已发生事实，或与供应链中断无关 |

**操作方式**：看 `llm_suggest`（warning/not_warning）和 `llm_reason` 作参考 → 同意就照抄，不同意就改判。`llm_suggest` 是辅助，**以你的判断为准**。

**只填 `human_label`，不要改其他列**（尤其 round1 的 52 行一个都不要动）。

## 三、核心判定标准

> **报道发出时，它所警告的供应链中断（trigger）还没有发生。**

判为 `1`（真预警）的例子：
- 报道明确指出"可能/或将/风险在于"某个尚未发生的供应链中断；
- 报道日期（`signal_date`）早于事件 onset，内容有实质预测。

判为 `0`（非预警）的例子：
- 报道描述的中断**已经发生**（误标为 forward）；
- 关键词渗漏（提到事件词但讲的是无关的事，如"日本地震风险"泛泛而谈）；
- 通用风险套话（"供应链面临不确定性"没有具体内容）。

## 四、完成后

68 行确认完，告诉我，我会：
1. 重算 §4.4 抽查精度（n=51 → n=120，Wilson CI 收窄）；
2. 更新论文对应段落；
3. 若新增 68 行里 `0` 占比明显，还会给每事件分层报告。

## 五、辅助信息

- 文件别名：`data/annotation/forward_spotcheck_v3.csv`（符号链接指向本文件，脚本兼容用）
- 生成脚本：`pipeline/extend_spotcheck_sample.py`（round1 52 行 + 新增 68 行）
- LLM 预标注：`pipeline/prefill_spotcheck_v3.py`（qwen3.6-27b 本地，temperature=0）
- round1 仲裁源：`forward_spotcheck_v2_adjudication.xlsx`（label_A/label_B/adjudicated）

## 六、事后剔除（标注完成后追加）

标注时样本为 120 行；论文口径为 **109 行**，剔除规则见
`pipeline/finalize_spotcheck_v3.py` 的 `EXCLUDE` 与 `EXCLUDE_EVENTS`：

| 剔除 | 行数 | 理由 |
|---|---|---|
| `us_china_tariff_war_2018-2017-03-24-0000-1` | 1 | GDELT 把 2018-03-24 的猪肉报道错标为 2017-03-24，信息重复且落在 180 天窗口外 |
| `toyota_steel_explosion_2019` 全部信号 | 10 | 该事件已从 `gt_events.json` 撤回（无任何来源可佐证，见 `pipeline/prune_gt_toyota.py`）；判断"是不是真预警"以该事件为前提，故不能作为抽查证据 |

这 10 条**全部是 protocol miss**，因此剔除只影响 miss 层与合并数，hit 层不变：

| | 120 行（旧） | 109 行（论文口径） |
|---|---|---|
| 合并精度 | 96/119 = 0.807 | **92/109 = 0.844** |
| hit 层 | 59/66 = 0.894 | 59/66 = 0.894（不变） |
| miss 层 | 37/53 = 0.698 | **33/43 = 0.767** |

重跑 `pipeline/finalize_spotcheck_v3.py` 即可从 `_d.csv` / `_z.xlsx` 重建这 109 行；
该脚本对已定稿的行是幂等的（重跑不会改动任何保留行的标签）。
