"""
app.py
Streamlit web app for Customer Review Insights.
"""

import os
import pandas as pd
import streamlit as st

from src.data_utils import load_reviews_csv, add_sentiment_predictions
from src.analytics import compute_sentiment_summary, sentiment_by_source


st.set_page_config(
    page_title="Customer Review Insights",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_sample_data():
    sample_path = os.path.join("data", "reviews_sample.csv")
    if not os.path.exists(sample_path):
        return None
    df = load_reviews_csv(sample_path)
    df = add_sentiment_predictions(df)
    return df


def main():
    st.title("📊 Customer Review Insights Dashboard")
    st.write(
        "Analyze customer reviews with an ML-based sentiment classifier "
        "and get quick insights for product and support teams."
    )

    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file with a 'review_text' column",
        type=["csv"],
    )

    use_sample = st.sidebar.checkbox("Use sample data (reviews_sample.csv)", value=True)

    df = None

    if uploaded_file is not None:
        # User uploaded their own CSV
        user_df = pd.read_csv(uploaded_file)
        df = add_sentiment_predictions(user_df)
        st.sidebar.success("Custom file loaded and predictions added.")
    elif use_sample:
        df = load_sample_data()
        if df is not None:
            st.sidebar.info("Using bundled sample dataset.")
        else:
            st.sidebar.error("Sample dataset not found.")
    else:
        st.sidebar.warning("Please upload a CSV or select sample data.")

    if df is None:
        st.stop()

    # Top-level KPIs
    summary = compute_sentiment_summary(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", summary["total_reviews"])
    col2.metric("Positive Reviews", f"{summary['positive_count']} ({summary['positive_pct']:.1f}%)")
    col3.metric("Negative Reviews", f"{summary['negative_count']} ({summary['negative_pct']:.1f}%)")

    st.markdown("---")

    # Charts and tables
    tab1, tab2, tab3 = st.tabs(["Overview", "By Source", "Raw Data"])

    with tab1:
        st.subheader("Sentiment Distribution")

        sentiment_counts = (
            df["sentiment_label"].value_counts().reindex(["positive", "negative"]).fillna(0)
        )
        st.bar_chart(sentiment_counts)

        st.subheader("High-confidence Negative Reviews (for Support)")
        neg_reviews = df[df["sentiment_label"] == "negative"].copy()
        neg_reviews = neg_reviews.sort_values("sentiment_score").head(10)
        if not neg_reviews.empty:
            st.dataframe(
                neg_reviews[["review_text", "sentiment_label", "sentiment_score", "rating"]]
                if "rating" in neg_reviews.columns
                else neg_reviews[["review_text", "sentiment_label", "sentiment_score"]]
            )
        else:
            st.write("No negative reviews found in this dataset.")

    with tab2:
        st.subheader("Sentiment by Source")
        source_stats = sentiment_by_source(df)
        if source_stats is not None:
            st.dataframe(source_stats)
        else:
            st.write("No 'source' column available in this dataset.")

    with tab3:
        st.subheader("Raw Data with Predictions")
        st.dataframe(df)


if __name__ == "__main__":
    main()
