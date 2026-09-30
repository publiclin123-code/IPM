# 检索审计标注指引 — 文章级相关/不相关 (v1.0)

> 版本 v1.0 · 标注对象:`data/annotation/retrieval_audit_sample.csv`
> 对应论文 §4.3 "Retrieval audit and recall bounds" 的 **Precision audit**。
> 目的:给 GDELT 关键词检索(文章池)做**纯度审计**,回答一个窄问题——
> "关键词过滤器捞回来的这些文章里,有多少篇真的跟它所归属的事件相关"。
> **本任务与提取层抽检(forward_spotcheck_v2)完全不同:**
> - 抽检标的是**信号**(LLM 抽取产物):"这条预警是真的吗"
> - 本任务标的是**文章**(检索产物):"这篇新闻跟这个事件相关吗"

---

## 一、你要做的只有一件事

对 CSV 每一行,填 `relevant_label` 列:

| 值 | 含义 | 判定 |
|----|------|------|
| `1` | **相关** | 这篇文章报道的内容,确实属于该事件(供应链中断/受影响商品/公司/地区) |
| `0` | **无关** | 文章与该事件不沾边,只是关键词碰巧重合(检索误报) |
| `?` | **存疑** | 拿不准(标题太模糊、正文为空、URL 已失效) |

`note` 列选填(如"关键词 false match 自 X 词"),其余列**不要改**。

---

## 二、核心判据

> **判断对象是「文章内容与该事件的相关性」,不是「文章质量」,更不是「LLM 抽得好不好」。**

一篇 `relevant=1` 的文章,应当满足**至少一条**:

1. **事件本身**:报道的是该事件的触发、进展、或直接后果(如红海危机 → 胡塞袭击/绕行/运价)。
2. **受影响商品/行业**:报道该事件所涉商品、行业、供应链环节的受影响情况。
3. **受影响公司/地区**:报道事件中受影响的公司(如 Toyota、TSMC)或地区(如熊本、台湾)。
4. **前兆/背景**:报道该事件发生前的诱因、前兆、或相关地缘/政策/疫情背景(只要与事件链条直接相关)。

一篇 `relevant=0` 的文章,典型是**关键词假阳性**:

- 事件关键词与无关新闻撞词(见下方真实例子)。
- 文章主题与该事件八竿子打不着(纯娱乐、体育、地方杂闻、无关司法)。

---

## 三、操作流程

1. 先看 `event_name`,明确这个事件是关于什么、涉及哪些商品/公司/地区。
2. 再看 `article_title` + `article_url`(URL 的 slug 常比标题信息更多)。
3. 再看 `article_text`(正文预览,多数是 slug 化标题,信息有限)。
4. 综合判断,填 `relevant_label`。

**优先级:标题/URL slug > 正文 > 事件名猜测。**

**注意**:GDELT 的 `title` 和 `text` 常常只是 URL 的 slug 化文本(去掉了 `-`、`_`),不是真正的新闻正文。判断主要靠**标题语义 + URL 域名/路径**。标题实在看不出、URL 也失效的,填 `?`。

---

## 四、事件清单(快速参考)

判断前先定位该行 `event_id` 属于哪个事件:

| event_id | 事件(一句话) |
|----------|--------------|
| red_sea_crisis_2023 | 红海航运危机(胡塞袭击商船) |
| suez_ever_given_2021 | 长赐号堵塞苏伊士运河 |
| us_chip_export_controls_2022 | 美国对华芯片出口管制 |
| covid_supply_disruption_2020 | 新冠疫情全球供应链中断 |
| renesas_earthquake_2016 | 熊本地震致瑞萨芯片厂停产 |
| port_los_angeles_backlog_2021 | 洛杉矶/长滩港拥堵 |
| toyota_steel_explosion_2019 | 丰田工厂爆炸停产 |
| europe_energy_crisis_2022 | 欧洲能源危机(俄气断供) |
| renesas_naka_plant_fire_2021 | 瑞萨那珂工厂火灾 |
| taiwan_strait_crisis_2022 | 台海危机(佩洛西访台/军演) |
| us_china_tariff_war_2018 | 美国对华 301 关税 |
| hurricane_maria_2017 | 飓风玛丽亚重创波多黎各(医药) |
| uaw_auto_strike_2023 | UAW 对底特律三巨头罢工 |
| black_sea_grain_exit_2023 | 俄退出黑海粮食协议 |
| india_wheat_export_ban_2022 | 印度热浪后禁小麦出口 |
| egg_shortage_birdflu_2025 | 美国禽流感致鸡蛋短缺涨价 |
| russia_ukraine_war_2022 | 俄乌全面战争 |
| beirut_port_explosion_2020 | 贝鲁特港硝酸铵爆炸 |

---

## 五、真实假阳性示例(务必体会)

这些是从文章池里实际看到的 `relevant=0` 例子,关键词重合的典型:

| 事件 | 文章标题 | 为什么无关 |
|------|---------|-----------|
| toyota_steel_explosion_2019 | "cow explosion black hole neutron star" | 天文新闻撞了 `explosion` |
| us_chip_export_controls_2022 | "ontario judge rejects cannabis impaired driving" | 大麻与 `chip`/半导体无关,纯地方司法 |
| us_chip_export_controls_2022 | "cannabis themed farmers market wisconsin" | 大麻农贸,与芯片无关 |
| warehouse_collapse_lithium_2019 | "forest fire in Jefferson County" | 山火撞了 `fire` 关键词 |
| toyota_steel_explosion_2019 | "Steelcase to Issue $450M Senior Notes" | 家具公司融资,撞了 `Steel` |

**相反,`relevant=1` 的正确例子:**

| 事件 | 文章标题 | 为什么相关 |
|------|---------|-----------|
| red_sea_crisis_2023 | "Iran allies threaten US over Israel intervention" | 红海/中东地缘前兆 |
| us_chip_export_controls_2022 | "bill to boost semiconductor industry passes Senate" | 半导体政策背景 |
| europe_energy_crisis_2022 | "Gazprom cuts Nord Stream gas flows" | 俄气断供直接相关 |

---

## 六、交付与后续

- 填好的 CSV:`relevant_label` 全部非空(`1` / `0` / `?`)。
- 单人约 400–450 篇,每篇只看标题/URL slug 判相关,预计 **30–60 分钟**。
- 完成后原样交付。我按事件汇总 per-event precision 与 Wilson 95% 区间,填入论文 §4.3 第一处 TODO。
- 论文口径:这是**检索纯度审计**(retrieval precision audit),不是提取层 gold set,也不涉及召回(召回下界是另一批标注,需第二个正交检索器)。
