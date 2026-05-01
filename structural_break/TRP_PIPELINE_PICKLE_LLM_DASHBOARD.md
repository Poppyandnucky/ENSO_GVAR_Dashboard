# TRP 全流程说明：pickle 生成、LLM、可视化与产出清单

这份文档用于跨会话/跨项目快速恢复上下文，覆盖：

- 阶段 A：生成 `gvar_pipeline_results.pkl`
- 阶段 B：基于 pickle 做可视化 dashboard
- 阶段 C：LLM 任务输入来源与相关输出

---

## 0. 主流程脚本（建议以这三个为主）

- `GVAR_LLM_pickle.py`：构建并保存 pipeline bundle（核心产物是 pickle）
- `Dashboard_visualize_from_pickle.py`：只读 pickle 并输出图件（不重新拟合模型）
- `LLM_ALL.py`：从 pickle 中提取 break 年份窗口，供文档抓取/大模型判别链路使用

> 备注：仓库里有 `GVAR_LLM_pikle.py`（拼写不同），功能相近但建议优先使用 `GVAR_LLM_pickle.py` 作为主线版本，避免混淆。

---

## 1. 阶段 A：生成 pickle（核心数据打包）

## 1.1 输入来源

在 `GVAR_LLM_pickle.py` 内，核心输入包括：

- 面板数据 CSV（由 `PATH` 指定）
- 国家列表（`countries_to_run`）
- 变量配置（`ENDO`、`EXO`、`lags`）
- 可选 LLM 结果 CSV（`LLM_RESULTS_CSV`，用于集成 break 文本标签）

## 1.2 处理内容（脚本内部做了什么）

脚本会构建 `pickle_bundle`，主要包含：

- `config`：运行配置、路径、国家范围、作图默认路径等
- `offline_plot_data["base"]`：每个国家的离线绘图数据（系数轨迹、诊断、Q/R trace）
- `llm_integration`（可选）：LLM 相关 DataFrame 与输入候选
- `refit` + `offline_plot_data["refit"]`（可选）：按 LLM 近断点删样本后的重估数据

## 1.3 关键输出文件（阶段 A）

默认输出到 `Dash_Input/`：

- `Dash_Input/gvar_pipeline_results.pkl`（最关键）
- `Dash_Input/input_LLM.pkl`（LLM 候选输入列表）
- `Dash_Input/llm_top5_near_break_points_to_drop.csv`（refit 删除点清单，若启用）
- `Dash_Input/gvar_panel_refit_filtered.csv`（refit 过滤后面板，若启用）

脚本中也定义了图路径（如 `GVAR_LLM_EM_plots.pdf`），但当前主策略是“先存 pickle，图由 dashboard 脚本统一生成”。

---

## 2. 阶段 B：从 pickle 可视化（Dashboard）

由 `Dashboard_visualize_from_pickle.py` 完成，默认读取：

- `PICKLE_PATH = "Dash_Input/gvar_pipeline_results.pkl"`

## 2.1 可视化模块

脚本分 3 大块，可独立开关：

- Core EM（`ENABLE_CORE_EM`）：系数轨迹 + 诊断 + Q/R 迭代轨迹
- LLM（`ENABLE_LLM`）：breakscore-LLM overlay + LLM 统计图 + 年度地图 HTML
- Refit（`ENABLE_REFIT`）：refit 后的 core 图与 breakscore 图

## 2.2 阶段 B 输出文件（按默认配置）

默认配置下（以脚本顶部为准）：

- Core EM：`pdf`
  - 输出：`Dash_Input/GVAR_LLM_EM_plots.pdf`（或 `config.PLOTS_PDF_PATH`）
- LLM overlay：`pdf`
  - 输出：`Dash_Input/gvar_breakscore_llm_overlay.pdf`（由 png 路径改后缀得到）
- LLM stats：`jpg`
  - 输出：`Dash_Input/gvar_llm_stats_ratio.jpg`
  - 输出：`Dash_Input/gvar_llm_stats_type.jpg`
- 时间切片地图：`on`
  - 输出目录：`Dash_Input/gvar_llm_time_slice_maps/`
  - 文件样式：`map_1998.html` ... `map_2015.html`
- Refit：默认关闭（`ENABLE_REFIT=False`），开启后会输出 refit 的 PDF/JPG

## 2.3 常用运行命令

```bash
python3 Dashboard_visualize_from_pickle.py
```

```bash
python3 Dashboard_visualize_from_pickle.py --pickle Dash_Input/gvar_pipeline_results.pkl
```

```bash
python3 Dashboard_visualize_from_pickle.py --countries CHL,MEX
```

---

## 3. 阶段 C：LLM 模型链路（与 A/B 相关）

这里分成“项目内可确定”与“外部模型调用”两层。

## 3.1 项目内可确定的 LLM 相关产物

来自 `GVAR_LLM_pickle.py` 的 `llm_integration`：

- `llm_df`：LLM 结果表（通常来自 `LLM_RESULTS_CSV` 映射后）
- `break_score_df`：EM 断点分数面板（quarter-level）
- `composite_break_df`：复合分数面板（含 z-score 组件）
- `input_llm_list`：供后续 LLM 查询的候选国家-年份列表
- `score_year_df`：按国家-年份聚合后的分数（地图输入）

这些都被塞入 `gvar_pipeline_results.pkl`，供 dashboard 直接用。

## 3.2 `LLM_ALL.py` 与模型调用输出

`LLM_ALL.py` 读取：

- `PIPELINE_RESULTS_PICKLE = "Dash_Input/gvar_pipeline_results.pkl"`

并根据 `BREAK_DICT_SOURCE` 从 pickle 的 `llm_integration` 提取 break 年份窗口。

脚本后段（当前文件内）包含文档抓取 + Gemini 调用流程，涉及输出：

- `wb_break_documents_english_exact_year.csv`
- `wb_document_level_scored.csv`
- `wb_top4.csv`
- `gemini output/*.csv`（每国家一个）
- `gemini_results_all_countries.xlsx`

同时可见模型配置示例：

- `MODEL_NAME = "gemini-3-flash-preview"`

> 注意：API key 与外部模型调用配置应在你本地安全管理，不建议硬编码在可共享仓库里。

---

## 4. 全部关键 output 一览（按用途）

- **主数据包**
  - `Dash_Input/gvar_pipeline_results.pkl`
- **LLM 输入候选**
  - `Dash_Input/input_LLM.pkl`
- **Core/LLM 图件**
  - `Dash_Input/GVAR_LLM_EM_plots.pdf`
  - `Dash_Input/gvar_breakscore_llm_overlay.pdf` 或 `.jpg`
  - `Dash_Input/gvar_llm_stats_ratio.jpg`
  - `Dash_Input/gvar_llm_stats_type.jpg`
- **地图**
  - `Dash_Input/gvar_llm_time_slice_maps/map_YYYY.html`
- **Refit 相关（启用时）**
  - `Dash_Input/llm_top5_near_break_points_to_drop.csv`
  - `Dash_Input/gvar_panel_refit_filtered.csv`
  - `Dash_Input/GVAR_LLM_EM_plots_refit.pdf`（或 refit JPG 目录）
- **外部文档+LLM 评估链路（LLM_ALL）**
  - `wb_break_documents_english_exact_year.csv`
  - `wb_document_level_scored.csv`
  - `wb_top4.csv`
  - `gemini output/*.csv`
  - `gemini_results_all_countries.xlsx`

---

## 5. 在 TRP_archive 迁移建议（最少 vs 完整）

## 5.1 最少可视化迁移

- `Dashboard_visualize_from_pickle.py`
- `llm_break_visualization.py`
- `Dash_Input/gvar_pipeline_results.pkl`

## 5.2 建议完整迁移（可复现更多环节）

- 上述最少集合 +
- `GVAR_LLM_pickle.py`
- `LLM_ALL.py`
- `Dash_Input/` 整体目录（包含历史图与中间文件）

---

## 6. 跨聊天记忆使用方式（推荐）

新会话第一句可直接写：

`先读取 TRP_PIPELINE_PICKLE_LLM_DASHBOARD.md，然后按文档在 TRP_archive 运行 pickle 生成和 dashboard 可视化。`

这样可以避免上下文丢失，快速恢复同一套流程。

