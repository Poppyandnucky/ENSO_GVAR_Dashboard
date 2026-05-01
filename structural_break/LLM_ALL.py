import hashlib
import pickle
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://search.worldbank.org/api/v3/wds"
OUTPUT_CSV = "wb_break_documents_english_exact_year.csv"

# Output of GVAR_LLM_pickle.py (same bundle as SDR_visualize).
PIPELINE_RESULTS_PICKLE = "Dash_Input/gvar_pipeline_results.pkl"
# Which slice of the pickle drives World Bank year windows:
#   "input_llm_candidates" — llm_integration["input_llm_list"] (composite peaks ±1;
#       same long list the pickle builds for LLM prompts; default).
#   "llm_df_all" — every break_year row in llm_df (merged LLM CSV).
#   "llm_df_supported" — llm_df rows where break_supported indicates yes (short list).
BREAK_DICT_SOURCE: str = "input_llm_candidates"


def _normalize_break_supported(s: pd.Series) -> pd.Series:
    """1 if row is LLM-supported break, else 0 (aligned with llm_break_visualization)."""

    def _one(x: object) -> int:
        if pd.isna(x):
            return 0
        t = str(x).strip().lower()
        if t in {"1", "true", "yes"}:
            return 1
        if t in {"0", "false", "no"}:
            return 0
        try:
            return 1 if int(float(x)) == 1 else 0
        except (TypeError, ValueError):
            return 0

    return s.map(_one).astype(int)


def _iso3_year_map_to_wb_break_dict(
    iso3_to_years: dict[str, list[int]],
    iso3_to_country: dict,
) -> dict[str, list[int]]:
    """Map ISO3 -> sorted unique years to World Bank ``count_exact`` country names."""
    out: dict[str, list[int]] = {}
    for iso3, years in iso3_to_years.items():
        iso = str(iso3).strip().upper()
        wb_name = iso3_to_country.get(iso)
        if wb_name is None:
            print(f"[LLM_ALL] WARN: ISO3={iso} not in iso3_to_country; skip.")
            continue
        ys = sorted({int(y) for y in years if pd.notna(y)})
        if not ys:
            continue
        key = str(wb_name).strip()
        if key in out:
            out[key] = sorted(set(out[key]) | set(ys))
        else:
            out[key] = ys
    return out


def load_break_dict_from_pipeline_pickle(
    pickle_path: str | Path,
    *,
    source: str | None = None,
) -> dict[str, list[int]]:

    p = Path(pickle_path)
    if not p.is_file():
        print(f"[LLM_ALL] WARN: pipeline pickle not found: {p.resolve()}")
        return {}

    with p.open("rb") as f:
        bundle = pickle.load(f)

    iso3_to_country: dict = bundle.get("config", {}).get("iso3_to_country") or {}
    llm = bundle.get("llm_integration") or {}
    mode = (source or BREAK_DICT_SOURCE).strip().lower()

    iso3_years: dict[str, list[int]] = {}

    if mode in ("input_llm_candidates", "candidates", "input_llm"):
        raw_list = llm.get("input_llm_list")
        if not raw_list:
            print(
                "[LLM_ALL] WARN: no input_llm_list in pickle; "
                "set BREAK_DICT_SOURCE to llm_df_all / llm_df_supported or rerun pickle."
            )
        else:
            for entry in raw_list:
                if not isinstance(entry, dict):
                    continue
                c = entry.get("country")
                ys = entry.get("years") or []
                if c is None:
                    continue
                iso3_years[str(c).strip().upper()] = [int(y) for y in ys if pd.notna(y)]
    elif mode in ("llm_df_all", "llm_df_supported", "supported", "all_llm"):
        llm_df = llm.get("llm_df")
        if llm_df is None or getattr(llm_df, "empty", True):
            print("[LLM_ALL] WARN: pickle has no llm_integration['llm_df'].")
            return {}

        df = llm_df.copy()
        if "break_year" not in df.columns or "country" not in df.columns:
            print("[LLM_ALL] WARN: llm_df missing break_year/country.")
            return {}

        df["break_year"] = pd.to_numeric(df["break_year"], errors="coerce")
        df = df.dropna(subset=["break_year"]).copy()
        df["break_year"] = df["break_year"].astype(int)

        if mode in ("llm_df_supported", "supported"):
            if "break_supported" not in df.columns:
                print("[LLM_ALL] WARN: llm_df has no break_supported; using all rows.")
            else:
                sup = _normalize_break_supported(df["break_supported"])
                df = df.loc[sup == 1].copy()

        if df.empty:
            print("[LLM_ALL] WARN: no llm_df rows left after filtering.")
            return {}

        for iso3, g in df.groupby("country"):
            iso3_years[str(iso3).strip().upper()] = g["break_year"].astype(int).tolist()
    else:
        print(
            f"[LLM_ALL] WARN: unknown BREAK_DICT_SOURCE={mode!r}; "
            "use input_llm_candidates | llm_df_all | llm_df_supported."
        )
        return {}

    out = _iso3_year_map_to_wb_break_dict(iso3_years, iso3_to_country)
    n_years = sum(len(v) for v in out.values())
    print(
        f"[LLM_ALL] Loaded break_dict (source={mode}): {len(out)} WB countries, {n_years} year-slots."
    )
    return out

# PRIORITY_DOCTYPES = [
#     "Annual Report",
#     "Country Economic Memorandum",
#     "Brief",
#     "Publication"
# ]

# FIELDS = [
#     "display_title",
#     "docdt",
#     "abstracts",
#     "pdfurl",
#     "txturl",
#     "url",
#     "docty",
#     "count",
#     "lang"
# ]


# def make_exact_year_window(year):
#     start_date = f"{year}-01-01"
#     end_date = f"{year}-12-31"
#     return start_date, end_date


# def build_query(country, year, docty, rows=50, os=0):
#     start_date, end_date = make_exact_year_window(year)

#     return {
#         "format": "json",
#         "count_exact": country,
#         "lang_exact": "English",
#         "strdate": start_date,
#         "enddate": end_date,
#         "docty_exact": docty,
#         "rows": rows,
#         "os": os,
#         "fl": ",".join(FIELDS),
#     }


# def fetch_page(params, timeout=30):
#     resp = requests.get(BASE_URL, params=params, timeout=timeout)
#     resp.raise_for_status()
#     return resp.json()


# def extract_documents(payload):
#     docs = payload.get("documents", {})
#     if not isinstance(docs, dict):
#         return []

#     rows = []
#     for k, v in docs.items():
#         if k == "facets":
#             continue
#         if not isinstance(v, dict):
#             continue

#         rows.append({
#             "wb_doc_id": k,
#             "display_title": v.get("display_title"),
#             "docdt": v.get("docdt"),
#             "abstracts": v.get("abstracts"),
#             "pdfurl": v.get("pdfurl"),
#             "txturl": v.get("txturl"),
#             "url": v.get("url"),
#             "docty": v.get("docty"),
#             "count": v.get("count"),
#             "lang": v.get("lang"),
#         })

#     return rows


# def fetch_all_docs_for_doctype(country, year, docty, rows_per_page=50, sleep_sec=0.2):
#     all_rows = []
#     offset = 0

#     while True:
#         params = build_query(
#             country=country,
#             year=year,
#             docty=docty,
#             rows=rows_per_page,
#             os=offset
#         )

#         payload = fetch_page(params)
#         page_rows = extract_documents(payload)

#         if not page_rows:
#             break

#         all_rows.extend(page_rows)

#         if len(page_rows) < rows_per_page:
#             break

#         offset += rows_per_page
#         time.sleep(sleep_sec)

#     return all_rows


# def make_llm_context(group_df):
#     lines = []
#     group_df = group_df.sort_values(["docdt", "display_title"]).reset_index(drop=True)

#     for i, row in group_df.iterrows():
#         title = row["display_title"] if pd.notna(row["display_title"]) else ""
#         docdt = row["docdt"] if pd.notna(row["docdt"]) else ""
#         docty = row["docty"] if pd.notna(row["docty"]) else ""
#         abstract = row["abstracts"] if pd.notna(row["abstracts"]) else ""
#         abstract = " ".join(str(abstract).split())

#         block = (
#             f"Document {i+1}\n"
#             f"Title: {title}\n"
#             f"Date: {docdt}\n"
#             f"Type: {docty}\n"
#             f"Abstract: {abstract}\n"
#         )
#         lines.append(block)

#     return "\n---\n".join(lines)


# def main():
#     break_dict = load_break_dict_from_pipeline_pickle(PIPELINE_RESULTS_PICKLE)
#     if not break_dict:
#         print("[LLM_ALL] break_dict is empty; nothing to fetch. Run GVAR_LLM_pickle.py with LLM CSV first.")
#         return

#     collected = []

#     for country, years in break_dict.items():
#         for year in years:
#             print(f"Processing: {country}, year={year}")

#             for docty in PRIORITY_DOCTYPES:
#                 try:
#                     docs = fetch_all_docs_for_doctype(country, year, docty)
#                 except Exception as e:
#                     print(f"Failed: {country}, {year}, {docty}, error={e}")
#                     continue

#                 for doc in docs:
#                     start_date, end_date = make_exact_year_window(year)

#                     collected.append({
#                         "country": country,
#                         "break_year": year,
#                         "window_start": start_date,
#                         "window_end": end_date,
#                         "searched_docty": docty,
#                         **doc
#                     })

#     out_df = pd.DataFrame(collected)

#     if out_df.empty:
#         print("No documents found.")
#         out_df.to_csv(OUTPUT_CSV, index=False)
#         return

#     out_df = out_df.drop_duplicates(
#         subset=["country", "break_year", "wb_doc_id", "display_title", "docdt"]
#     ).reset_index(drop=True)

#     llm_rows = []
#     for (country, break_year), g in out_df.groupby(["country", "break_year"], sort=False):
#         llm_rows.append(
#             {
#                 "country": country,
#                 "break_year": break_year,
#                 "llm_context": make_llm_context(g),
#                 "n_docs": len(g),
#             }
#         )
#     llm_df = pd.DataFrame(llm_rows)

#     final_df = out_df.merge(llm_df, on=["country", "break_year"], how="left")
#     final_df.to_csv(OUTPUT_CSV, index=False)

#     print(f"Saved to {OUTPUT_CSV}")
#     print(f"Total rows: {len(final_df)}")
#     print(f"Unique break points: {final_df[['country', 'break_year']].drop_duplicates().shape[0]}")


# if __name__ == "__main__":
#     main()


# import pandas as pd
# import re
# import ast

# INPUT_CSV = "wb_break_documents_english_exact_year.csv"
# OUTPUT_SCORED_CSV = "wb_document_level_scored.csv"
# OUTPUT_TOP_CSV = "wb_top4.csv"

# POSITIVE_KEYWORDS = {
#      # --- Crisis / downturn ---
#     "crisis": 3,
#     "financial crisis": 6,
#     "banking crisis": 6,
#     "banking instability": 6,
#     "global recession": 6,
#     "recession": 5,
#     "downturn": 4,
#     "economic contraction": 5,
#     "growth slowdown": 5,
#     "food crisis": 5,
#     # --- External / trade ---
#     "trade": 3,
#     "trade policy": 3,
#     "trade performance": 4,
#     "terms of trade": 5,
#     "export": 4,
#     "exports": 4,
#     "import": 4,
#     "imports": 4,
#     "global demand": 4,
#     "external shock": 5,
#     "balance of payments": 5,
#     "current account": 4,
#     # --- Prices / inflation ---
#     "inflation": 5,
#     "inflation shock": 5,
#     "commodity": 3,
#     "commodity prices": 5,
#     "oil prices": 4,
#     # --- Financial / risk ---
#     "debt": 4,
#     "public debt": 4,
#     "sovereign risk": 5,
#     "creditworthiness": 4,
#     "capital flows": 5,
#     # --- Policy ---
#     "fiscal policy": 5,
#     "fiscal adjustment": 5,
#     "monetary policy": 5,
#     "interest rate": 4,
#     "central bank": 4,
#     "policy reform": 4,
#     "structural reform": 4,
#     "institutional reform": 3,
#     "privatization": 3,
#     # --- Public sector ---
#     "public finance": 4,
#     "public expenditure": 3,
#     "budget": 3,
#     # --- Growth / macro ---
#     "economic growth": 3,
#     "macroeconomic": 4,
#     "macroeconomic stability": 5,
#     # --- Labor / social ---
#     "unemployment": 3,
#     "labor market": 2,
#     "inequality": 2,
#     "social protection": 2,
#     # --- Exchange rate ---
#     "exchange rate": 5,
#     "exchange rate depreciation": 6,
#     # --- Recovery ---
#     "recovery": 2,
#     # --- ENSO / climate system ---
#     "enso": 7,
#     "el nino": 7,
#     "la nina": 7,
#     "southern oscillation": 6,
#     "nino": 6,
#     # --- Extreme weather ---
#     "drought": 6,
#     "flood": 6,
#     "flooding": 6,
#     "heatwave": 6,
#     "extreme weather": 6,
#     "climate shock": 6,
#     "weather shock": 6,
#     "natural disaster": 6,
#     # --- Agriculture impact ---
#     "crop failure": 7,
#     "harvest loss": 6,
#     "agricultural output": 5,
#     "agricultural production": 5,
#     "food supply": 5,
#     "food shortage": 6,
#     # --- Commodity channel ---
#     "commodity shock": 6,
#     "food prices": 6,
#     "agricultural prices": 5,
#     # --- Water / environment ---
#     "water shortage": 6,
#     "water stress": 5,
#     # --- Climate policy ---
#     "climate policy": 4,
#     "climate adaptation": 4,
#     "climate mitigation": 4,
# }

# NEGATIVE_KEYWORDS = {
#     "project": -3,
#     "projects": -3,
#     "implementation": -2,
#     "implementation completion": -3,
#     "disbursement": -3,
#     "disbursements": -3,
#     "loan": -2,
#     "loans": -2,
#     "grant": -2,
#     "grants": -2,
#     "appraisal": -2,
#     "execution": -2,
#     "procurement": -3,
#     "safeguards": -3,
#     "resettlement": -3,
#     "municipal development": -2,
#     "highway": -2,
#     "irrigation": -2,
#     "water sector": -2,
#     "wastewater": -2,
#     "teacher": -3,
#     "teachers": -3,
#     "education": -1,
#     "school": -2,
#     "schools": -2,
#     "hospital": -2,
#     "hospitals": -2,
#     "congress": -1,
#     "legislators": -1,
#     "technology to train": -2,
#     "basic health": -2,
#     "small towns": -2,
#     "farm": -1,
#     "farming": -1,
#     "rural development": -2
# }

# DOCTYPE_WEIGHTS = {
#     "Country Economic Memorandum": 4,
#     "Annual Report": 2,
#     "Publication": 2,
#     "Journal Article": 2,
#     "Policy Research Working Paper": 1,
#     "Working Paper": 1,
#     "Working Paper (Numbered Series)": 1,
#     "Brief": 0
# }

# EMPTY_ABSTRACT_PENALTY = -6
# TOP_K = 4


# def extract_abstract_text(x):
#     if pd.isna(x):
#         return ""
#     x = str(x).strip()
#     try:
#         parsed = ast.literal_eval(x)
#         if isinstance(parsed, dict):
#             if "cdata!" in parsed:
#                 return str(parsed["cdata!"]).strip()
#             return " ".join(str(v) for v in parsed.values()).strip()
#     except Exception:
#         pass
#     return x


# def clean_text(x):
#     if pd.isna(x):
#         return ""
#     return " ".join(str(x).lower().split())


# def keyword_hits(text, keyword_dict):
#     score = 0
#     hits = []
#     for kw, wt in keyword_dict.items():
#         pattern = r"\b" + re.escape(kw.lower()) + r"\b"
#         if re.search(pattern, text):
#             score += wt
#             hits.append(kw)
#     return score, hits


# df = pd.read_csv(INPUT_CSV)

# required_cols = ["display_title", "docdt", "abstracts", "docty"]
# for col in required_cols:
#     if col not in df.columns:
#         raise ValueError(f"Missing required column: {col}")

# df["abstract_text"] = df["abstracts"].apply(extract_abstract_text)
# df["abstract_text_clean"] = df["abstract_text"].apply(clean_text)

# pos = df["abstract_text_clean"].apply(lambda x: keyword_hits(x, POSITIVE_KEYWORDS))
# neg = df["abstract_text_clean"].apply(lambda x: keyword_hits(x, NEGATIVE_KEYWORDS))

# df["positive_score"] = pos.apply(lambda x: x[0])
# df["positive_hits"] = pos.apply(lambda x: ", ".join(x[1]))

# df["negative_score"] = neg.apply(lambda x: x[0])
# df["negative_hits"] = neg.apply(lambda x: ", ".join(x[1]))

# df["doctype_score"] = df["docty"].map(DOCTYPE_WEIGHTS).fillna(0)
# df["empty_abstract_penalty"] = df["abstract_text"].apply(
#     lambda x: EMPTY_ABSTRACT_PENALTY if not str(x).strip() else 0
# )

# df["final_score"] = (
#     df["positive_score"]
#     + df["negative_score"]
#     + df["doctype_score"]
#     + df["empty_abstract_penalty"]
# )

# sort_cols = []
# ascending = []

# if "country" in df.columns:
#     sort_cols.append("country")
#     ascending.append(True)

# if "break_year" in df.columns:
#     sort_cols.append("break_year")
#     ascending.append(True)

# sort_cols += ["final_score", "docdt"]
# ascending += [False, True]

# df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

# df.to_csv(OUTPUT_SCORED_CSV, index=False)

# if "country" in df.columns and "break_year" in df.columns:
#     top_df = (
#         df.groupby(["country", "break_year"], group_keys=False)
#         .head(TOP_K)
#         .reset_index(drop=True)
#     )
# else:
#     top_df = df.head(TOP_K).copy()

# top_df.to_csv(OUTPUT_TOP_CSV, index=False)

# print(f"Saved full scored file: {OUTPUT_SCORED_CSV}")
# print(f"Saved top file: {OUTPUT_TOP_CSV}")
# print()
# show_cols = [
#     c for c in [
#         "country", "break_year", "display_title", "docty",
#         "final_score", "positive_score", "negative_score",
#         "doctype_score", "empty_abstract_penalty",
#         "positive_hits", "negative_hits", "abstract_text"
#     ] if c in top_df.columns
# ]
# print(top_df[show_cols].head(20).to_string(index=False))


import time
import pandas as pd
from google import genai

INPUT_CSV = "wb_top4.csv"

API_KEY = "AIzaSyATPLxgBM4x1rv782LAY3oO3G_hBwvK2dk"
MODEL_NAME = "gemini-3-flash-preview"
SLEEP_SECONDS = 2.0
OUTPUT_WORKBOOK = "gemini_results_all_countries.xlsx"
# One-country-per-csv output folder. If a country's csv already exists, skip that
# country entirely (both preprocess and Gemini calls) to avoid repeated token cost.
GEMINI_OUTPUT_DIR = Path("gemini output")

# Same scope as World Bank fetch: pickle + BREAK_DICT_SOURCE (file top). WB English names -> years.
TARGET_COUNTRIES = load_break_dict_from_pipeline_pickle(PIPELINE_RESULTS_PICKLE)
if not TARGET_COUNTRIES:
    raise RuntimeError(
        "TARGET_COUNTRIES is empty: fix PIPELINE_RESULTS_PICKLE / BREAK_DICT_SOURCE or run "
        "GVAR_LLM_pickle.py before the Gemini block."
    )

REQUIRED_COLUMNS = [
    "country",
    "break_year",
    "display_title",
    "docdt",
    "docty",
    "abstract_text"
]


def _excel_sheet_slug(target_country: str) -> str:
    s = (
        target_country.replace(":", "_")
        .replace("\\", "_")
        .replace("/", "_")
        .replace("?", "_")
        .replace("*", "_")
        .replace("[", "_")
        .replace("]", "_")
    )[:31]
    return s or "sheet"


def _gemini_country_csv_path(target_country: str) -> Path:
    base = (
        target_country.replace(":", "_")
        .replace("\\", "_")
        .replace("/", "_")
        .replace("?", "_")
        .replace("*", "_")
        .replace("[", "_")
        .replace("]", "_")
        .strip()
    )
    if not base:
        base = hashlib.md5(target_country.encode("utf-8")).hexdigest()[:12]
    return GEMINI_OUTPUT_DIR / f"{base}.csv"


def clean_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())

def build_document_block(row, doc_num):
    title = clean_text(row.get("display_title", ""))
    docdt = clean_text(row.get("docdt", ""))
    docty = clean_text(row.get("docty", ""))
    abstract = clean_text(row.get("abstract_text", ""))
    score = row.get("final_score", "")

    score_line = ""
    if pd.notna(score) and str(score).strip() != "":
        score_line = f"Score: {score}\n"

    return (
        f"Document {doc_num}\n"
        f"Title: {title}\n"
        f"Date: {docdt}\n"
        f"Type: {docty}\n"
        f"{score_line}"
        f"Abstract: {abstract}\n"
    )

def build_prompt(group_df):
    country = clean_text(group_df["country"].iloc[0])
    break_year = clean_text(group_df["break_year"].iloc[0])

    docs = []
    group_df = group_df.reset_index(drop=True)

    for i, (_, row) in enumerate(group_df.iterrows(), start=1):
        docs.append(build_document_block(row, i))

    docs_block = "\n---\n".join(docs)

    prompt = f"""You are helping identify whether a statistically detected structural break in macroeconomic data has plausible textual support from World Bank documents.

Country: {country}
Break year: {break_year}

Task:
Based only on the document summaries below, answer the following questions.

You MUST choose answers only from the given options.

Questions:

1. break_supported:
Choose one:
- 1
- 0
- -99

2. break_type:
Choose 1-2 from:
- financial_crisis
- policy_change
- external_shock
- climate_shock
- commodity_or_trade

3. duration:
Choose one:
- within 1 year
- 2 year
- long_term

4. climate_related:
Choose one:
- 1
- 0
- -99

Rules:
- Use ONLY the provided documents
- Do not infer beyond the text
- If evidence is weak, choose "Unclear"

Return EXACTLY in this format:

break_supported: 
break_type: 
duration: 
climate_related: 
confidence: 1-5
summary: [<10 key words]

Documents:
{docs_block}
"""
    return prompt

# ===== load data =====
df = pd.read_csv(INPUT_CSV)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in REQUIRED_COLUMNS:
    df[col] = df[col].apply(clean_text)

client = genai.Client(api_key=API_KEY)

# ----- Phase 1: run per country and save one csv per country -----
GEMINI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
excel_jobs: list[tuple[str, Path]] = []  # (sheet_name, country_csv)

for target_country, years in TARGET_COUNTRIES.items():

    print(f"\n=== Processing {target_country} ===")

    country_csv = _gemini_country_csv_path(target_country)
    safe_sheet = _excel_sheet_slug(target_country)

    # Hard skip by country: existing csv means no further work for this country.
    if country_csv.is_file():
        print(f"Skip country (existing csv): {country_csv.name}")
        excel_jobs.append((safe_sheet, country_csv))
        continue

    sub_df = df[df["country"] == target_country].copy()
    sub_df = sub_df[sub_df["break_year"].astype(str).isin([str(y) for y in years])]

    if sub_df.empty:
        print(f"No data for {target_country}")
        continue

    if "final_score" in sub_df.columns:
        sub_df = sub_df.sort_values(
            ["country", "break_year", "final_score", "docdt", "display_title"],
            ascending=[True, True, False, True, True],
        )
    else:
        sub_df = sub_df.sort_values(
            ["country", "break_year", "docdt", "display_title"],
            ascending=[True, True, True, True],
        )

    sub_df = sub_df.reset_index(drop=True)

    results: list[dict] = []

    for (country, break_year), group in sub_df.groupby(["country", "break_year"], sort=True):
        prompt = build_prompt(group)

        status = "ok"
        error_message = ""
        raw_output = ""

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            raw_output = response.text if hasattr(response, "text") else str(response)

        except Exception as e:
            status = "error"
            error_message = repr(e)

        results.append(
            {
                "country": country,
                "break_year": break_year,
                "n_docs": len(group),
                "status": status,
                "error_message": error_message,
                "raw_output": raw_output,
            }
        )
        pd.DataFrame(results).to_csv(country_csv, index=False)

        print(f"Done: {country}, {break_year}, status={status} -> {country_csv.name}")
        time.sleep(SLEEP_SECONDS)

    if country_csv.is_file() and country_csv.stat().st_size > 0:
        excel_jobs.append((safe_sheet, country_csv))

# ----- Phase 2: merge per-country csv files into one workbook -----
used_names: set[str] = set()

with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
    any_sheet = False
    for safe_sheet, country_csv in excel_jobs:
        out_df = pd.read_csv(country_csv)
        if out_df.empty:
            continue
        sheet = safe_sheet
        base = sheet
        n = 2
        while sheet in used_names:
            suffix = f"_{n}"
            sheet = (base[: 31 - len(suffix)] + suffix)[:31]
            n += 1
        used_names.add(sheet)
        out_df.to_excel(writer, sheet_name=sheet, index=False)
        any_sheet = True
        print(f"Sheet -> {sheet} from {country_csv.name}")

    if not any_sheet:
        pd.DataFrame(
            {
                "note": [
                    "No country CSV to export. "
                    "Check wb_top4 vs TARGET_COUNTRIES or remove country CSVs to rerun."
                ]
            }
        ).to_excel(writer, sheet_name="_no_data", index=False)
        print("[WARN] Wrote placeholder sheet _no_data (no non-empty country CSVs).")

print(f"\nSaved workbook: {OUTPUT_WORKBOOK}")
print(f"Country CSVs under: {GEMINI_OUTPUT_DIR.resolve()}")