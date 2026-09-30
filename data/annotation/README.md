# 人工标注目录索引

本目录按标注任务组织，每个活跃任务一个文件夹，含专属说明 md。

## 活跃任务（待你操作）

| 文件夹 | 任务 | 待标行数 | 说明文件 |
|---|---|---|---|
| `task1_retrieval_audit_body/` | 正文池检索审计（文章相关性） | **348** | `README_标注指引.md` |
| `task2_spotcheck_v3/` | 信号抽查扩容确认（预警语义） | **68** | `README_标注指引.md` |

## 已完成 / 归档

| 文件 | 状态 | 说明 |
|---|---|---|
| `work2/` | ✅ 已完成 | 标题池检索审计 389 行，双盲仲裁终版 `final_labels_merged.csv` |
| `forward_spotcheck_v2_adjudication.xlsx` | ✅ 已完成 | round1 信号抽查 52 条仲裁表（label_A/label_B/adjudicated） |
| `forward_spotcheck_v2.csv` / `v2_round2.csv` | 归档 | 第一/二轮盲标样本（52 条），已被 v3 取代 |
| `retrieval_audit_sample.csv` | 归档 | 标题池检索审计样本（389 行，标题版），已被 body 版取代 |
| `_archive_v1_forward_signals_annotation.csv` | 归档 | v1 时代旧样本，弃用 |
| `retrieval_audit_sample_body.csv.bak2` | 备份 | 回填前原始版 |

## 路径兼容说明

`retrieval_audit_sample_body.csv` 和 `forward_spotcheck_v3.csv` 在 `data/annotation/` 根目录是**符号链接**，指向各自任务文件夹内的真实文件。这样 11 个 pipeline 脚本无需改动路径即可读写同一份数据。

## 论文对应关系

- §4.3 Retrieval audit and recall bounds ← `task1_retrieval_audit_body/` + `work2/`
- §4.4 Models and evaluation design (spot-check) ← `task2_spotcheck_v3/` + `forward_spotcheck_v2_adjudication.xlsx`
