
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Survey Dashboard with Filter Validation", layout="wide")
st.title("Survey Thematic Analysis Dashboard")
st.caption("Deterministic TF-IDF + NMF thematic pipeline with local NRC emotion scoring and live filter validation")

DEFAULT_FILE = Path(__file__).parent / "icecream_full_pipeline_with_filter_validation.xlsx"
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]

@st.cache_data(show_spinner=False)
def load_workbook(file_or_path):
    xl = pd.ExcelFile(file_or_path)
    return {sheet: pd.read_excel(file_or_path, sheet_name=sheet) for sheet in xl.sheet_names}

def add_numeric_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    created = {}
    for col in list(out.columns):
        if col == "response_id":
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        non_null = numeric.dropna()
        if non_null.empty or non_null.nunique() < 10:
            continue
        new_col = f"{col}_group"
        if col.lower() == "age":
            bins = [0, 17, 24, 34, 44, 54, 64, float("inf")]
            labels = ["Under 18", "18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 to 64", "65 or over"]
            out[new_col] = pd.cut(numeric, bins=bins, labels=labels, right=True, include_lowest=True)
        else:
            try:
                out[new_col] = pd.qcut(numeric, q=5, duplicates="drop").astype(str).replace("nan", pd.NA)
            except ValueError:
                out[new_col] = pd.cut(numeric, bins=5, include_lowest=True).astype(str).replace("nan", pd.NA)
        created[col] = new_col
    return out, created

def safe_pattern(values):
    return "|".join(re.escape(str(v)) for v in values)

def traffic_light(value, green, amber):
    if value >= green:
        return "High", "✅"
    if value >= amber:
        return "Moderate", "⚠️"
    return "Low", "🚨"

def compute_filter_validation(responses, assignments, emotions, filtered_responses):
    total_n = len(responses)
    filtered_n = len(filtered_responses)
    filtered_ids = set(filtered_responses["response_id"]) if "response_id" in filtered_responses.columns else set()

    if filtered_n == 0:
        return {
            "score": 0,
            "label": "Very Low",
            "warnings": ["No responses remain after filtering."],
            "metrics": {
                "Filtered responses": 0,
                "Segment share": 0,
                "Topic coverage": 0,
                "NRC coverage": 0,
                "Small themes": 0,
            },
        }

    stage1 = assignments[
        assignments["assignment_stage"].eq("Stage 1")
        & assignments["response_id"].isin(filtered_ids)
    ].copy()

    assigned_unique = stage1["response_id"].nunique()
    topic_coverage = assigned_unique / max(filtered_n, 1)
    segment_share = filtered_n / max(total_n, 1)

    if not emotions.empty and "nrc_coverage_rate" in emotions.columns:
        nrc_coverage = emotions[emotions["response_id"].isin(filtered_ids)]["nrc_coverage_rate"].mean()
        if pd.isna(nrc_coverage):
            nrc_coverage = 0
    else:
        nrc_coverage = 0

    topic_counts = (
        stage1.groupby(["topic_id", "topic_name"])["response_id"]
        .nunique()
        .reset_index(name="count")
    )
    small_theme_count = int((topic_counts["count"] < 5).sum()) if not topic_counts.empty else 0

    # Score components: response count 30, topic coverage 25, NRC 15, segment share 15, small-theme quality 15.
    response_score = 30 if filtered_n >= 100 else 20 if filtered_n >= 25 else 5
    topic_score = 25 if topic_coverage >= 0.80 else 15 if topic_coverage >= 0.60 else 5
    nrc_score = 15 if nrc_coverage >= 0.05 else 8 if nrc_coverage >= 0.02 else 2
    segment_score = 15 if segment_share >= 0.05 else 8 if segment_share >= 0.01 else 2
    small_theme_score = 15 if small_theme_count == 0 else 8 if small_theme_count <= 2 else 2
    score = int(response_score + topic_score + nrc_score + segment_score + small_theme_score)

    if score >= 85:
        label = "High"
    elif score >= 70:
        label = "Moderate"
    elif score >= 50:
        label = "Low"
    else:
        label = "Very Low"

    warnings = []
    if filtered_n < 25:
        warnings.append(f"Only {filtered_n} responses are in the current filter context.")
    elif filtered_n < 100:
        warnings.append(f"{filtered_n} responses are in the current filter context; interpret subgroup patterns cautiously.")

    if topic_coverage < 0.60:
        warnings.append(f"Topic coverage is low at {topic_coverage:.1%}.")
    elif topic_coverage < 0.80:
        warnings.append(f"Topic coverage is moderate at {topic_coverage:.1%}.")

    if nrc_coverage < 0.02:
        warnings.append(f"NRC token coverage is very low at {nrc_coverage:.1%}.")
    elif nrc_coverage < 0.05:
        warnings.append(f"NRC token coverage is moderate at {nrc_coverage:.1%}.")

    if segment_share < 0.01:
        warnings.append(f"This segment represents only {segment_share:.1%} of all responses.")
    elif segment_share < 0.05:
        warnings.append(f"This segment represents {segment_share:.1%} of all responses.")

    if small_theme_count:
        warnings.append(f"{small_theme_count} visible parent topics contain fewer than 5 assigned responses.")

    return {
        "score": score,
        "label": label,
        "warnings": warnings,
        "metrics": {
            "Filtered responses": filtered_n,
            "Segment share": segment_share,
            "Topic coverage": topic_coverage,
            "NRC coverage": nrc_coverage,
            "Small themes": small_theme_count,
        },
        "topic_counts": topic_counts,
    }

def filtered_topic_counts(assignments, filtered_ids, stage, denominator):
    if assignments.empty or not filtered_ids:
        return pd.DataFrame(columns=["topic_id", "topic_name", "count", "percent"])
    df = assignments[
        assignments["response_id"].isin(filtered_ids)
        & assignments["assignment_stage"].eq(stage)
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=["topic_id", "topic_name", "count", "percent"])
    if stage == "Stage 1":
        grouped = df.groupby(["topic_id", "topic_name"], dropna=False)["response_id"].nunique().reset_index(name="count")
        grouped["percent"] = grouped["count"] / max(denominator, 1)
        return grouped.sort_values(["count", "topic_name"], ascending=[False, True])
    grouped = df.groupby(["topic_id", "topic_name", "subtheme_id", "subtheme_name"], dropna=False)["response_id"].nunique().reset_index(name="count")
    grouped["percent"] = grouped["count"] / max(denominator, 1)
    return grouped.sort_values(["count", "subtheme_name"], ascending=[False, True])

def emotion_distribution(emotions, filtered_ids, denominator):
    if emotions.empty or not filtered_ids or "dominant_emotion" not in emotions.columns:
        return pd.DataFrame(columns=["dominant_emotion", "count", "percent"])
    view = emotions[emotions["response_id"].isin(filtered_ids)]
    grouped = view["dominant_emotion"].fillna("none").value_counts().rename_axis("dominant_emotion").reset_index(name="count")
    grouped["percent"] = grouped["count"] / max(denominator, 1)
    return grouped

def metric_bar(df, x, y, title, orientation="v", color=None):
    if df.empty:
        st.info("No data available for the current filters.")
        return
    view = df.copy()
    pct_col = "percent" if "percent" in view.columns else None
    if pct_col:
        view["label"] = view["count"].fillna(0).astype(int).astype(str) + " (" + view[pct_col].fillna(0).map(lambda v: f"{v:.1%}") + ")"
    else:
        view["label"] = view["count"].fillna(0).astype(int).astype(str)
    fig = px.bar(view, x=x, y=y, color=color, orientation=orientation, text="label", title=title)
    if orientation == "h":
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(r=140))
    else:
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

def validation_score_card(validation: dict):
    """Render the filter validation score using a traffic-light colour system."""
    label = validation.get("label", "Very Low")
    score = int(validation.get("score", 0))
    if label == "High":
        colour = "#2E7D32"
        icon = "●"
    elif label == "Moderate":
        colour = "#F9A825"
        icon = "●"
    else:
        colour = "#C62828"
        icon = "●"

    st.sidebar.markdown(
        f"""
        <div style="
            border-left: 8px solid {colour};
            background: rgba(128, 128, 128, 0.08);
            padding: 0.75rem 0.85rem;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;">
            <div style="font-size: 0.85rem; opacity: 0.8;">Validation score</div>
            <div style="font-size: 1.7rem; font-weight: 700; color: {colour}; line-height: 1.2;">
                {icon} {score}/100
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: {colour};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_representative_context_columns(responses: pd.DataFrame) -> list[str]:
    """Detect respondent context fields to display beside representative verbatims.

    Includes age fields and location-style variables such as country, region,
    market, city, state, county, province, territory, geography, or location.
    Original response text remains unchanged.
    """
    if responses.empty:
        return []

    excluded = {
        "response_id",
        "original_response",
        "assigned_parent_topics",
        "assigned_parent_topic_names",
        "assigned_subthemes",
        "assigned_subtheme_names",
    }
    location_terms = (
        "country",
        "region",
        "uk_region",
        "market",
        "location",
        "geo",
        "geography",
        "city",
        "town",
        "county",
        "state",
        "province",
        "territory",
        "postcode",
        "postal",
        "area",
        "district",
    )

    detected = []
    for col in responses.columns:
        col_key = str(col).strip().lower()
        if col in excluded or col_key.startswith("openbox_"):
            continue
        if col_key == "age" or col_key.endswith("_age") or col_key == "age_group":
            detected.append(col)
            continue
        if any(term in col_key for term in location_terms):
            detected.append(col)

    # Stable de-duplication while preserving dataset column order.
    return list(dict.fromkeys(detected))


def format_context_value(value) -> str:
    """Format metadata values for representative verbatim display."""
    if pd.isna(value):
        return "Not provided"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

def representative_verbatims(responses, assignments, filtered_ids, stage, topic_name=None, subtheme_name=None, max_n=5):
    """Return the most representative verbatims for a selected chart element.

    Representative = highest cosine similarity assignment score within the
    current dashboard filter context. Verbatim text is preserved unchanged.
    """
    required = {"response_id", "assignment_stage", "similarity_score"}
    if responses.empty or assignments.empty or not filtered_ids or not required.issubset(assignments.columns):
        return pd.DataFrame(columns=["response_id", "similarity_score", "original_response"])

    view = assignments[
        assignments["response_id"].isin(filtered_ids)
        & assignments["assignment_stage"].eq(stage)
    ].copy()

    if topic_name and "topic_name" in view.columns:
        view = view[view["topic_name"].astype(str).eq(str(topic_name))]
    if subtheme_name and "subtheme_name" in view.columns:
        view = view[view["subtheme_name"].astype(str).eq(str(subtheme_name))]

    if view.empty:
        return pd.DataFrame(columns=["response_id", "similarity_score", "original_response"])

    response_cols = ["response_id", "original_response"]
    context_cols = detect_representative_context_columns(responses)
    for optional_col in context_cols + ["dominant_emotion"]:
        if optional_col in responses.columns and optional_col not in response_cols:
            response_cols.append(optional_col)

    out = (
        view.sort_values(["similarity_score", "response_id"], ascending=[False, True])
        .merge(responses[response_cols], on="response_id", how="left")
        .drop_duplicates("response_id")
        .head(max_n)
    )

    keep_cols = ["response_id", "similarity_score", "original_response"]
    keep_cols += [c for c in context_cols + ["dominant_emotion"] if c in out.columns and c not in keep_cols]
    return out[keep_cols]

def selectable_metric_bar(df, x, y, title, selector_key, orientation="h", color=None):
    """Render a bar chart with point selection enabled and return selected label."""
    if df.empty:
        st.info("No data available for the current filters.")
        return None

    view = df.copy()
    pct_col = "percent" if "percent" in view.columns else None
    if pct_col:
        view["label"] = view["count"].fillna(0).astype(int).astype(str) + " (" + view[pct_col].fillna(0).map(lambda v: f"{v:.1%}") + ")"
    else:
        view["label"] = view["count"].fillna(0).astype(int).astype(str)

    # The selected field is always carried as custom data for reliable lookup.
    selected_field = y if orientation == "h" else x
    fig = px.bar(
        view,
        x=x,
        y=y,
        color=color,
        orientation=orientation,
        text="label",
        title=title,
        custom_data=[selected_field],
    )
    if orientation == "h":
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(r=140))
    else:
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_tickangle=-35)

    selected_label = None
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=selector_key,
            on_select="rerun",
            selection_mode="points",
        )
        points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
        if points and points[0].get("customdata"):
            selected_label = points[0]["customdata"][0]
    except TypeError:
        # Fallback for older Streamlit versions without chart selection support.
        st.plotly_chart(fig, use_container_width=True, key=selector_key)

    fallback_options = view[selected_field].dropna().astype(str).tolist()
    fallback_options = sorted(set(fallback_options))
    manual_selection = st.selectbox(
        "Select a chart section to inspect representative verbatims",
        [""] + fallback_options,
        key=f"{selector_key}_manual_selection",
    )
    if manual_selection:
        selected_label = manual_selection

    return selected_label


def selected_points_from_plotly(fig, selector_key):
    """Render a selectable Plotly figure and return selected point dictionaries."""
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=selector_key,
            on_select="rerun",
            selection_mode="points",
        )
        return event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=selector_key)
        return []


def representative_emotion_verbatims(responses, emotions, filtered_ids, emotion_label=None, emotion_rate_col=None, max_n=5):
    """Return representative verbatims for emotion charts within the active filter context."""
    if responses.empty or emotions.empty or not filtered_ids:
        return pd.DataFrame(columns=["response_id", "score", "original_response"])

    view = emotions[emotions["response_id"].isin(filtered_ids)].copy()

    score_col = None
    if emotion_rate_col and emotion_rate_col in view.columns:
        score_col = emotion_rate_col
        view = view[pd.to_numeric(view[score_col], errors="coerce").fillna(0) > 0]
    elif emotion_label and "dominant_emotion" in view.columns:
        view = view[view["dominant_emotion"].fillna("none").astype(str).eq(str(emotion_label))]
        if "dominant_emotion_score" in view.columns:
            score_col = "dominant_emotion_score"
        elif "nrc_coverage_rate" in view.columns:
            score_col = "nrc_coverage_rate"

    if view.empty:
        return pd.DataFrame(columns=["response_id", "score", "original_response"])

    if score_col:
        view["score"] = pd.to_numeric(view[score_col], errors="coerce").fillna(0)
    else:
        view["score"] = 0

    response_cols = ["response_id", "original_response"]
    context_cols = detect_representative_context_columns(responses)
    for optional_col in context_cols + ["dominant_emotion"]:
        if optional_col in responses.columns and optional_col not in response_cols:
            response_cols.append(optional_col)

    out = (
        view.sort_values(["score", "response_id"], ascending=[False, True])
        .merge(responses[response_cols], on="response_id", how="left")
        .drop_duplicates("response_id")
        .head(max_n)
    )

    keep_cols = ["response_id", "score", "original_response"]
    keep_cols += [c for c in context_cols + ["dominant_emotion"] if c in out.columns and c not in keep_cols]
    return out[keep_cols]


def show_emotion_representative_verbatims(selection_label, responses, emotions, filtered_ids, emotion_rate_col=None):
    if not selection_label:
        st.info("Click a bar in the chart, or use the selector above, to show representative verbatims.")
        return

    reps = representative_emotion_verbatims(
        responses=responses,
        emotions=emotions,
        filtered_ids=filtered_ids,
        emotion_label=selection_label if not emotion_rate_col else None,
        emotion_rate_col=emotion_rate_col,
        max_n=5,
    )

    st.subheader(f"Top representative verbatims: {selection_label}")
    if reps.empty:
        st.info("No representative verbatims are available for this selection.")
        return

    context_cols = [c for c in reps.columns if c not in {"response_id", "score", "original_response"}]
    for _, row in reps.iterrows():
        context_html = ""
        if context_cols:
            context_items = [
                f"<span><strong>{col}:</strong> {format_context_value(row[col])}</span>"
                for col in context_cols
            ]
            context_html = (
                "<div style='font-size: 0.85rem; opacity: 0.80; margin: 0.25rem 0 0.45rem 0;'>"
                + " · ".join(context_items)
                + "</div>"
            )
        st.markdown(
            f"""
            <div style="border: 1px solid rgba(128,128,128,0.25); border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; opacity: 0.75;">
                    {row['response_id']} · score {row['score']:.3f}
                </div>
                {context_html}
                <div style="font-size: 1rem;">{row['original_response']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def selectable_segment_topic_bar(grouped, seg_col, selector_key):
    """Render selectable segment-by-topic chart and return selected segment/topic."""
    if grouped.empty:
        st.info("No segment data available.")
        return None, None

    fig = px.bar(
        grouped,
        x=seg_col,
        y="percent",
        color="topic_name",
        barmode="group",
        title=f"Parent topic coverage by {seg_col}",
        custom_data=[seg_col, "topic_name"],
    )
    fig.update_yaxes(tickformat=".0%")

    selected_segment, selected_topic = None, None
    points = selected_points_from_plotly(fig, selector_key)
    if points and points[0].get("customdata"):
        selected_segment = points[0]["customdata"][0]
        selected_topic = points[0]["customdata"][1]

    options = (
        grouped[[seg_col, "topic_name"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values([seg_col, "topic_name"])
    )
    manual_options = [""] + [f"{r[seg_col]} | {r['topic_name']}" for _, r in options.iterrows()]
    manual_selection = st.selectbox(
        "Select a chart section to inspect representative verbatims",
        manual_options,
        key=f"{selector_key}_manual_selection",
    )
    if manual_selection:
        selected_segment, selected_topic = manual_selection.split(" | ", 1)

    return selected_segment, selected_topic


def show_segment_representative_verbatims(selected_segment, selected_topic, seg_col, filtered_responses, responses, assignments):
    if not selected_segment or not selected_topic:
        st.info("Click a segment chart bar, or use the selector above, to show representative verbatims.")
        return

    segment_ids = set(
        filtered_responses.loc[
            filtered_responses[seg_col].astype(str).eq(str(selected_segment)),
            "response_id",
        ]
    )
    reps = representative_verbatims(
        responses,
        assignments,
        segment_ids,
        stage="Stage 1",
        topic_name=selected_topic,
        max_n=5,
    )

    st.subheader(f"Top representative verbatims: {selected_segment} · {selected_topic}")
    if reps.empty:
        st.info("No representative verbatims are available for this selection.")
        return

    context_cols = [c for c in reps.columns if c not in {"response_id", "similarity_score", "original_response"}]
    for _, row in reps.iterrows():
        context_html = ""
        if context_cols:
            context_items = [
                f"<span><strong>{col}:</strong> {format_context_value(row[col])}</span>"
                for col in context_cols
            ]
            context_html = (
                "<div style='font-size: 0.85rem; opacity: 0.80; margin: 0.25rem 0 0.45rem 0;'>"
                + " · ".join(context_items)
                + "</div>"
            )
        st.markdown(
            f"""
            <div style="border: 1px solid rgba(128,128,128,0.25); border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; opacity: 0.75;">
                    {row['response_id']} · similarity {row['similarity_score']:.3f}
                </div>
                {context_html}
                <div style="font-size: 1rem;">{row['original_response']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_representative_verbatims(selection_label, stage, responses, assignments, filtered_ids):
    if not selection_label:
        st.info("Click a bar in the chart, or use the selector above, to show representative verbatims.")
        return

    if stage == "Stage 1":
        reps = representative_verbatims(
            responses,
            assignments,
            filtered_ids,
            stage="Stage 1",
            topic_name=selection_label,
            max_n=5,
        )
        st.subheader(f"Top representative verbatims: {selection_label}")
    else:
        reps = representative_verbatims(
            responses,
            assignments,
            filtered_ids,
            stage="Stage 2",
            subtheme_name=selection_label,
            max_n=5,
        )
        st.subheader(f"Top representative verbatims: {selection_label}")

    if reps.empty:
        st.info("No representative verbatims are available for this selection.")
        return

    context_cols = [
        c for c in reps.columns
        if c not in {"response_id", "similarity_score", "original_response"}
    ]

    for _, row in reps.iterrows():
        context_html = ""
        if context_cols:
            context_items = [
                f"<span><strong>{col}:</strong> {format_context_value(row[col])}</span>"
                for col in context_cols
            ]
            context_html = (
                "<div style='font-size: 0.85rem; opacity: 0.80; margin: 0.25rem 0 0.45rem 0;'>"
                + " · ".join(context_items)
                + "</div>"
            )

        st.markdown(
            f"""
            <div style="border: 1px solid rgba(128,128,128,0.25); border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; opacity: 0.75;">
                    {row['response_id']} · similarity {row['similarity_score']:.3f}
                </div>
                {context_html}
                <div style="font-size: 1rem;">{row['original_response']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

uploaded = st.sidebar.file_uploader("Upload analysis Excel export", type=["xlsx"])
path = uploaded if uploaded else DEFAULT_FILE
data = load_workbook(path)

responses = data.get("Responses", pd.DataFrame()).copy()
assignments = data.get("Topic_Assignments", pd.DataFrame()).copy()
emotions = data.get("NRC_Emotion_Analysis", pd.DataFrame()).copy()
topics = data.get("Topics", pd.DataFrame()).copy()
audit = data.get("Audit_Log", pd.DataFrame()).copy()
validation_rules = data.get("Validation_Rules", pd.DataFrame()).copy()

responses, grouped_numeric_map = add_numeric_groups(responses)

raw_exclude = {
    "response_id", "original_response", "assigned_parent_topics", "assigned_parent_topic_names",
    "assigned_subthemes", "assigned_subtheme_names", "dominant_emotion", "dominant_emotion_score",
    "nrc_coverage_rate"
}
structured_cols = []
if not responses.empty:
    original_numeric_cols = set(grouped_numeric_map.keys())
    for col in responses.columns:
        if col in raw_exclude or col.startswith("openbox_") or col in original_numeric_cols:
            continue
        if responses[col].dtype == "object" or str(responses[col].dtype).startswith("category"):
            if responses[col].nunique(dropna=True) <= max(50, len(responses) * 0.5):
                structured_cols.append(col)
if "age_group" in responses.columns and "age_group" not in structured_cols:
    structured_cols.insert(0, "age_group")
structured_cols = [c for c in structured_cols if c != "age"]

st.sidebar.header("Filters")
filtered_responses = responses.copy()

for col in structured_cols:
    vals = sorted([v for v in filtered_responses[col].dropna().astype(str).unique()])
    chosen = st.sidebar.multiselect(col, vals)
    if chosen:
        filtered_responses = filtered_responses[filtered_responses[col].astype(str).isin(chosen)]

if "dominant_emotion" in filtered_responses.columns:
    vals = sorted([v for v in filtered_responses["dominant_emotion"].dropna().astype(str).unique()])
    chosen = st.sidebar.multiselect("Dominant emotion", vals)
    if chosen:
        filtered_responses = filtered_responses[filtered_responses["dominant_emotion"].astype(str).isin(chosen)]

if "assigned_parent_topic_names" in filtered_responses.columns:
    parent_vals = sorted(set(", ".join(filtered_responses["assigned_parent_topic_names"].fillna("").astype(str)).split(", ")) - {""})
    chosen = st.sidebar.multiselect("Stage 1 parent topic", parent_vals)
    if chosen:
        filtered_responses = filtered_responses[filtered_responses["assigned_parent_topic_names"].fillna("").str.contains(safe_pattern(chosen), regex=True)]

if "assigned_subtheme_names" in filtered_responses.columns:
    sub_vals = sorted(set(", ".join(filtered_responses["assigned_subtheme_names"].fillna("").astype(str)).split(", ")) - {""})
    chosen = st.sidebar.multiselect("Stage 2 sub-theme", sub_vals)
    if chosen:
        filtered_responses = filtered_responses[filtered_responses["assigned_subtheme_names"].fillna("").str.contains(safe_pattern(chosen), regex=True)]

validation = compute_filter_validation(responses, assignments, emotions, filtered_responses)

st.sidebar.divider()
st.sidebar.subheader("Filter validation")
validation_score_card(validation)

m = validation["metrics"]
st.sidebar.write(f"Responses: **{m['Filtered responses']}**")
st.sidebar.write(f"Segment share: **{m['Segment share']:.1%}**")
st.sidebar.write(f"Topic coverage: **{m['Topic coverage']:.1%}**")
st.sidebar.write(f"NRC coverage: **{m['NRC coverage']:.1%}**")
st.sidebar.write(f"Small themes: **{m['Small themes']}**")

if validation["warnings"]:
    with st.sidebar.expander("Validation warnings", expanded=True):
        for warning in validation["warnings"]:
            st.warning(warning)
else:
    st.sidebar.success("No validation warnings for the current filter context.")

filtered_ids = set(filtered_responses["response_id"]) if "response_id" in filtered_responses.columns else set()
denominator = len(filtered_responses)
parent_counts = filtered_topic_counts(assignments, filtered_ids, "Stage 1", denominator)
subtheme_counts = filtered_topic_counts(assignments, filtered_ids, "Stage 2", denominator)
emotion_counts = emotion_distribution(emotions, filtered_ids, denominator)

tabs = st.tabs(["Executive Overview", "Topic Analysis", "Sub-Theme Drilldown", "Segment Analysis", "Response Explorer", "Emotion Analysis", "QA & Audit"])

with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total responses", len(responses))
    c2.metric("Filtered responses", denominator)
    c3.metric("Validation score", f"{validation['score']}/100")
    c4.metric("Topic coverage", f"{m['Topic coverage']:.1%}")
    c5.metric("NRC coverage", f"{m['NRC coverage']:.1%}")

    if validation["warnings"]:
        st.warning("Current filter context has validation warnings. Review the sidebar before interpreting charts.")

    st.subheader("Parent topic distribution")
    metric_bar(parent_counts.head(15), x="count", y="topic_name", orientation="h", title="Parent topics within current filters")

    st.subheader("Dominant emotion distribution")
    metric_bar(emotion_counts, x="dominant_emotion", y="count", title="Dominant NRC emotion within current filters")

with tabs[1]:
    st.subheader("Topic Analysis")
    selected_topic = selectable_metric_bar(
        parent_counts,
        x="count",
        y="topic_name",
        orientation="h",
        title="Filtered parent topic frequency",
        selector_key="topic_analysis_parent_chart",
    )
    show_representative_verbatims(selected_topic, "Stage 1", responses, assignments, filtered_ids)
    st.dataframe(parent_counts, use_container_width=True)

with tabs[2]:
    st.subheader("Sub-Theme Drilldown")
    selected_subtheme = selectable_metric_bar(
        subtheme_counts.head(30),
        x="count",
        y="subtheme_name",
        orientation="h",
        color="topic_name" if "topic_name" in subtheme_counts.columns else None,
        title="Filtered sub-theme frequency",
        selector_key="subtheme_drilldown_chart",
    )
    show_representative_verbatims(selected_subtheme, "Stage 2", responses, assignments, filtered_ids)
    st.dataframe(subtheme_counts, use_container_width=True)

with tabs[3]:
    st.subheader("Segment Analysis")
    if structured_cols:
        seg_col = st.selectbox("Structured variable", structured_cols)
        st.caption("The sidebar validation score updates when this segment is further filtered.")
        base = filtered_responses[["response_id", seg_col]].dropna()
        view = assignments[assignments["response_id"].isin(filtered_ids) & assignments["assignment_stage"].eq("Stage 1")].merge(base, on="response_id", how="inner")
        if not view.empty:
            denom = base.groupby(seg_col)["response_id"].nunique().rename("segment_total").reset_index()
            grouped = view.groupby([seg_col, "topic_name"])["response_id"].nunique().reset_index(name="count").merge(denom, on=seg_col)
            grouped["percent"] = grouped["count"] / grouped["segment_total"]
            selected_segment, selected_topic = selectable_segment_topic_bar(
                grouped,
                seg_col=seg_col,
                selector_key="segment_analysis_topic_chart",
            )
            show_segment_representative_verbatims(
                selected_segment,
                selected_topic,
                seg_col,
                filtered_responses,
                responses,
                assignments,
            )
            st.dataframe(grouped, use_container_width=True)
        else:
            st.info("No segment data available.")
    else:
        st.info("No structured variables detected.")

with tabs[4]:
    st.subheader("Response Explorer")
    search = st.text_input("Search response text")
    view = filtered_responses.copy()
    if search and "original_response" in view.columns:
        view = view[view["original_response"].fillna("").str.contains(search, case=False, regex=False)]
    st.dataframe(view, use_container_width=True)

with tabs[5]:
    st.subheader("Emotion Analysis")

    selected_dominant_emotion = selectable_metric_bar(
        emotion_counts,
        x="dominant_emotion",
        y="count",
        title="Filtered dominant NRC emotion distribution",
        selector_key="emotion_dominant_chart",
        orientation="v",
    )
    show_emotion_representative_verbatims(
        selected_dominant_emotion,
        responses,
        emotions,
        filtered_ids,
    )

    if not emotions.empty and filtered_ids:
        filtered_emotions = emotions[emotions["response_id"].isin(filtered_ids)].copy()
        rate_cols = [f"{e}_rate" for e in EMOTIONS if f"{e}_rate" in filtered_emotions.columns]
        if rate_cols:
            avg_rates = filtered_emotions[rate_cols].mean().reset_index()
            avg_rates.columns = ["emotion", "average_rate"]
            avg_rates["emotion"] = avg_rates["emotion"].str.replace("_rate", "", regex=False)
            avg_rates["count"] = (avg_rates["average_rate"] * denominator).round(0).astype(int)
            avg_rates["percent"] = avg_rates["average_rate"]
            selected_avg_emotion = selectable_metric_bar(
                avg_rates.sort_values("average_rate", ascending=False),
                x="emotion",
                y="average_rate",
                title="Average NRC emotion token rate",
                selector_key="emotion_average_rate_chart",
                orientation="v",
            )
            if selected_avg_emotion:
                show_emotion_representative_verbatims(
                    selected_avg_emotion,
                    responses,
                    emotions,
                    filtered_ids,
                    emotion_rate_col=f"{selected_avg_emotion}_rate",
                )
        st.dataframe(filtered_emotions, use_container_width=True)

with tabs[6]:
    st.subheader("Current Filter Validation")
    st.json({k: v for k, v in validation.items() if k != "topic_counts"})
    st.subheader("Validation Rules")
    st.dataframe(validation_rules, use_container_width=True)
    st.subheader("Audit Log")
    st.dataframe(audit, use_container_width=True)
    st.subheader("Stage Summary")
    st.dataframe(data.get("Stage_Summary", pd.DataFrame()), use_container_width=True)
    st.subheader("Column Classification")
    st.dataframe(data.get("Column_Classification", pd.DataFrame()), use_container_width=True)

with st.sidebar.expander("Dashboard notes"):
    st.write("The validation score is recalculated after every filter change.")
    st.write("Scores are diagnostic quality indicators, not statistical confidence intervals.")
    if grouped_numeric_map:
        st.write("Grouped numeric filters:")
        st.json(grouped_numeric_map)
