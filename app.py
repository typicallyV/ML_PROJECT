"""Streamlit app - Social Media Engagement Prediction.
Run:  streamlit run app.py      (needs social_data.py in the same folder)
"""
from pathlib import Path
import calendar
from datetime import date
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from social_data import load_raw, build_xy, make_row, TYPES, WEEKDAYS

st.set_page_config(page_title="Social Media Engagement Predictor", page_icon="📱", layout="wide")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def get_data():
    df, src = load_raw(verbose=False)
    X, y_reg, y_clf, thr = build_xy(df)
    return df, src, X, y_reg, y_clf, thr


@st.cache_resource(show_spinner="Training models...")
def get_models():
    _, _, X, y_reg, y_clf, _ = get_data()
    model_path = BASE_DIR / "social_media_models.joblib"
    if model_path.exists():        # produced by the notebook (tuned params)
        try:
            m = joblib.load(model_path)
            if list(m["columns"]) == list(X.columns):
                return m["clf"], m["reg"], "Tuned models from Colab notebook"
        except Exception:
            pass
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42).fit(X, y_clf)
    reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y_reg)
    return clf, reg, "Trained inside the app (Random Forest)"


@st.cache_data(show_spinner="Comparing classifiers...")
def compare_models():
    _, _, X, _, y_clf, _ = get_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y_clf, test_size=0.2, stratify=y_clf, random_state=42)
    models = {
        "Logistic Regression": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
        "KNN (k=7)": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(7))]),
        "Naive Bayes": Pipeline([("s", StandardScaler()), ("m", GaussianNB())]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }
    rows = []
    for n, m in models.items():
        p = m.fit(Xtr, ytr).predict(Xte)
        rows.append({"Model": n, "Accuracy": accuracy_score(yte, p),
                     "Precision": precision_score(yte, p, zero_division=0),
                     "Recall": recall_score(yte, p, zero_division=0), "F1": f1_score(yte, p, zero_division=0)})
    return pd.DataFrame(rows).set_index("Model").round(3)


df, source, X, y_reg, y_clf, THR = get_data()
clf, reg, model_note = get_models()

st.title("📱 Social Media Engagement Predictor")
st.caption(f"Data: {source} · Models: {model_note}")
st.markdown(
    f"High engagement means at least **{THR:,.0f} total interactions**, "
    "the 75th percentile in the available dataset. Predictions are estimates "
    "from historical patterns and do not guarantee future results."
)
if source.startswith("SYNTHETIC"):
    st.warning(
        "The UCI dataset could not be loaded, so this session is using synthetic "
        "sample data. Predictions and model comparisons are for demonstration only. "
        "Add dataset_Facebook.csv beside app.py to use the real dataset offline."
    )

with st.sidebar:
    st.header("Describe your post")
    post_type = st.selectbox("Post type", TYPES, index=1)
    category = st.selectbox("Content category", [1, 2, 3])
    page_likes = st.slider("Page total likes", 70000, 150000, 120000, step=1000)
    planned_date = st.date_input("Planned publish date", value=date.today())
    month = planned_date.month
    weekday = ((planned_date.weekday() + 1) % 7) + 1
    hour = st.slider("Post hour (1-23)", 1, 23, 11)
    paid = st.radio("Paid promotion?", [0, 1], format_func=lambda v: "Yes" if v else "No", horizontal=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict", "⏰ Best time to post", "🧪 Compare post ideas", "📊 Model comparison", "🗂 Data explorer"
])

with tab1:
    row = make_row(X.columns, page_likes, post_type, category, month, weekday, hour, paid)
    prob = float(clf.predict_proba(row)[0][1])
    est = float(reg.predict(row)[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Chance of HIGH engagement", f"{prob*100:.1f}%")
    c2.metric("Predicted class", "High 🔥" if prob >= 0.5 else "Normal")
    c3.metric("Estimated interactions", f"{est:.0f}")
    st.progress(min(max(prob, 0.0), 1.0))
    day_name = WEEKDAYS[weekday]
    paid_label = "with paid promotion" if paid else "without paid promotion"
    prediction_label = "high engagement" if prob >= 0.5 else "typical engagement"
    st.info(
        f"**Quick summary:** Your {post_type.lower()} post in category {category}, "
        f"planned for **{day_name}, {planned_date.strftime('%B %d, %Y')} at {hour:02d}:00**, "
        f"{paid_label}, has a **{prob:.0%}** estimated chance of high engagement "
        f"({prediction_label}) and about **{est:,.0f} total interactions**."
    )
    st.caption("Change the post type dropdown, publish date, time, or other sidebar controls to update the prediction and summary.")
    if hasattr(clf, "feature_importances_"):
        st.subheader("What drives engagement? (feature importance)")
        imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        imp.plot.barh(ax=ax, color="#1D7874")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

with tab2:
    st.write(f"Probability of high engagement for a **{post_type}** post, by weekday and hour (other inputs from the sidebar).")
    st.caption("These are model estimates based on the selected inputs; they show patterns in the dataset, not causal effects.")
    grid = [make_row(X.columns, page_likes, post_type, category, month, d, h, paid)
            for d in WEEKDAYS for h in range(1, 24)]
    G = pd.concat(grid, ignore_index=True)
    G["p"] = clf.predict_proba(G)[:, 1]
    piv = G.pivot_table(index="Post Weekday", columns="Post Hour", values="p")
    piv.index = [WEEKDAYS[i] for i in piv.index]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    heatmap = ax.imshow(piv.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(range(len(piv.index)), labels=piv.index)
    ax.set_xticks(range(len(piv.columns)), labels=piv.columns)
    ax.set_xlabel("Post hour")
    ax.set_ylabel("Weekday")
    fig.colorbar(heatmap, ax=ax, label="P(high engagement)")
    st.pyplot(fig)
    top_n = st.slider("Number of suggested slots", 3, 10, 5, key="top_slots")
    best = G.sort_values("p", ascending=False).head(top_n)[["Post Weekday", "Post Hour", "p"]].copy()
    best["Post Weekday"] = best["Post Weekday"].map(WEEKDAYS)
    best["p"] = (best["p"] * 100).round(1).astype(str) + "%"
    st.write(f"Top {top_n} slots for these selected post details:")
    st.table(best.rename(columns={"p": "P(high)"}).reset_index(drop=True))

with tab3:
    st.write("Edit the post ideas below, add rows, then compare their predictions side by side.")
    default_scenarios = pd.DataFrame([
        {"Post idea": "Current plan", "Type": post_type, "Category": category, "Month": month,
         "Weekday": WEEKDAYS[weekday], "Hour": hour, "Paid": "Yes" if paid else "No", "Page likes": page_likes},
        {"Post idea": "Photo at lunch", "Type": "Photo", "Category": 1, "Month": month,
         "Weekday": WEEKDAYS[weekday], "Hour": 12, "Paid": "No", "Page likes": page_likes},
        {"Post idea": "Promoted status", "Type": "Status", "Category": 2, "Month": month,
         "Weekday": WEEKDAYS[weekday], "Hour": 17, "Paid": "Yes", "Page likes": page_likes},
    ])
    with st.form("scenario_comparison"):
        scenarios = st.data_editor(
            default_scenarios,
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Type": st.column_config.SelectboxColumn("Post type", options=TYPES, required=True),
                "Category": st.column_config.SelectboxColumn("Category", options=[1, 2, 3], required=True),
                "Month": st.column_config.SelectboxColumn(
                    "Month", options=list(range(1, 13)), format_func=lambda m: calendar.month_name[m], required=True
                ),
                "Weekday": st.column_config.SelectboxColumn("Weekday", options=list(WEEKDAYS.values()), required=True),
                "Paid": st.column_config.SelectboxColumn("Paid promotion", options=["No", "Yes"], required=True),
                "Hour": st.column_config.NumberColumn("Hour (0–23)", min_value=0, max_value=23, step=1, required=True),
                "Page likes": st.column_config.NumberColumn("Page likes", min_value=0, step=1000, required=True),
            },
            key="scenario_editor",
        )
        run_comparison = st.form_submit_button("Compare post ideas", type="primary")

    if run_comparison:
        if scenarios.empty:
            st.warning("Add at least one post idea to compare.")
        else:
            valid = scenarios.dropna(subset=["Post idea", "Type", "Category", "Month", "Weekday", "Hour", "Paid", "Page likes"])
            if valid.empty:
                st.warning("Complete at least one row before comparing.")
            else:
                scenario_rows = [make_row(
                    X.columns, int(r["Page likes"]), r["Type"], int(r["Category"]), int(r["Month"]),
                    {v: k for k, v in WEEKDAYS.items()}[r["Weekday"]], int(r["Hour"]), int(r["Paid"] == "Yes")
                ) for _, r in valid.iterrows()]
                scenario_X = pd.concat(scenario_rows, ignore_index=True)
                results = valid[["Post idea", "Type", "Weekday", "Hour", "Paid"]].reset_index(drop=True).copy()
                results["Chance of high engagement"] = (clf.predict_proba(scenario_X)[:, 1] * 100).round(1)
                results["Estimated interactions"] = np.maximum(0, reg.predict(scenario_X)).round().astype(int)
                results = results.sort_values("Estimated interactions", ascending=False).reset_index(drop=True)
                st.dataframe(results, hide_index=True, use_container_width=True)
                st.bar_chart(results.set_index("Post idea")[["Estimated interactions"]])
                st.download_button(
                    "Download comparison as CSV", results.to_csv(index=False).encode("utf-8"),
                    file_name="post_idea_comparison.csv", mime="text/csv"
                )

with tab4:
    st.write("Same 80:20 stratified split for every model (see Colab notebook for full practical workflow).")
    cmp_df = compare_models()
    st.dataframe(cmp_df.style.highlight_max(axis=0, color="#cdeee9"))
    st.bar_chart(cmp_df)

with tab5:
    st.write(f"Explore the {df.shape[0]} posts × {df.shape[1]} columns in the active dataset.")
    f1, f2, f3 = st.columns(3)
    type_filter = f1.multiselect("Filter by post type", TYPES, default=TYPES)
    category_filter = f2.multiselect("Filter by category", sorted(df["Category"].dropna().unique().tolist()))
    month_filter = f3.multiselect("Filter by month", list(range(1, 13)), format_func=lambda m: calendar.month_name[m])
    filtered = df[df["Type"].isin(type_filter)]
    if category_filter:
        filtered = filtered[filtered["Category"].isin(category_filter)]
    if month_filter:
        filtered = filtered[filtered["Post Month"].isin(month_filter)]
    st.caption(f"Showing {len(filtered)} of {len(df)} posts")
    filtered_display = filtered.copy()
    filtered_display["Engagement"] = np.where(
        filtered_display["Total Interactions"] >= THR, "High", "Normal"
    )
    st.dataframe(filtered_display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_facebook_posts.csv", mime="text/csv"
    )
    cc = filtered_display["Engagement"].value_counts()
    st.write("Class balance")
    st.bar_chart(cc)
    st.write("Summary statistics")
    st.dataframe(filtered.describe().T)

