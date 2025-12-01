import streamlit as st
import sys
import os

# Make sure we can import from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.predict import predict_text  # now this should work


def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered"
    )

    st.title("📰 Fake News Detection App")
    st.write(
        "Enter a news headline or short article text below and the model "
        "will predict whether it is **FAKE** or **REAL**."
    )

    user_input = st.text_area(
        "News text:",
        height=150,
        placeholder="Type or paste a news headline / short article here..."
    )

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                prediction = predict_text(user_input)
                if prediction.upper() == "FAKE":
                    st.error("⚠️ This looks like **FAKE** news (according to the model).")
                else:
                    st.success("✅ This looks **REAL** (according to the model).")

                st.caption(
                    "Note: This is a demo ML model trained on a small dataset. "
                    "Predictions are not perfect and should not be treated as facts."
                )
            except Exception as e:
                st.error(f"An error occurred while making prediction: {e}")


if __name__ == "__main__":
    main()
