# 检索审计（正文池）标注指引 — 文章级相关/不相关

> 版本 v2.0 · 标注对象：`task1_retrieval_audit_body/retrieval_audit_sample_body.csv`（389 行）
> 对应论文 §4.3 "Retrieval audit and recall bounds" 的 **Precision audit（正文版）**。
> 本任务由旧的标题池审计（work2，389 行已仲裁）升级而来：正文回收后工作表已回填
> 真文，现在要基于**文章正文**重新判断相关性。

---

## 一、当前进度

| 状态 | 行数 | 说明 |
|---|---|---|
| 已填 `relevant_label_body` | 41 | 机械关闭行（挑战页无正文）保留的标签 |
| **待你标注** | **348** | `relevant_label_body` 为空的行 |

## 二、你要做的只有一件事

对 CSV 每一行，填 **`relevant_label_body`** 列：

| 值 | 含义 |
|----|------|
| `1` | 相关：这篇文章报道的内容确实属于该事件（供应链中断/受影响商品/公司/地区） |
| `0` | 无关：文章与该事件不沾边，只是关键词碰巧重合 |
| `?` | 存疑：拿不准 |

**只填这一列**，其余列不要改（`llm_suggest`/`llm_reason` 是 LLM 预标注参考，可参考但以你的判断为准）。

## 三、核心判据

> **判断对象是「文章正文内容与该事件的相关性」，不是文章质量，更不是 LLM 抽得好不好。**

1. 看 `article_body`（已回填的真文）；`body_status` 为 `title_fallback` 的 41 行没有正文，保持原标签不动；
2. 关键词碰巧重合（如"toyota"出现在无关新闻里）→ `0`；
3. 正文描述了该事件的前兆、发展、影响、应对、相关公司/商品/地区 → `1`；
4. 正文是事后报道（事件已发生，报道损失/追责）→ 仍算**相关**（§4.3 审计的是检索相关性，不是时间方向）。

## 四、完成后

标完 348 行后运行：

```bash
python pipeline/compare_annotations.py
```

我会据此把论文 §4.3 的 pooled precision 更新到正文池版本。

## 五、辅助信息

- 文件别名：`data/annotation/retrieval_audit_sample_body.csv`（符号链接指向本文件，脚本兼容用）
- 备份：`retrieval_audit_sample_body.csv.bak2`（回填前原始版）
- 生成脚本：`pipeline/build_audit_body_worksheet.py`
- LLM 预标注：`pipeline/prefill_annotation_body.py`
