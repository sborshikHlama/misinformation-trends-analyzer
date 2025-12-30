import os
import json

import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Style Detector")
st.markdown(
    "Enter news text — the model will tell you if it's written in a "
    "fake news style and what topic it belongs to"
)


def check_api_health() -> dict:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "unhealthy", "models_loaded": False}


def analyze_text(text: str) -> dict:
    response = requests.post(f"{API_URL}/analyze", json={"text": text}, timeout=60)
    response.raise_for_status()
    return response.json()


# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    health = check_api_health()
    if health.get("status") == "healthy":
        st.success("✅ API Connected")
        st.write(f"**BERTopic:** {'✅' if health.get('bertopic_loaded') else '❌'}")
        st.write(f"**LogReg:** {'✅' if health.get('logreg_loaded') else '❌'}")
    else:
        st.error("❌ API Disconnected")
        st.write(f"API URL: `{API_URL}`")

    st.divider()
    st.header("📚 Model Information")
    st.write("**BERTopic:** Topic modeling")
    st.write("**LogReg:** Fake/Real classification")

# Main input
user_text = st.text_area(
    "Enter news text:",
    height=200,
    placeholder="For example: Scientists discover shocking new evidence...",
)

if st.button("Analyze Text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter text to analyze!")
    elif health.get("status") != "healthy":
        st.error("Backend API is not available.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = analyze_text(user_text)
                st.success("Analysis complete!")

                topic = result["topic"]
                classification = result["classification"]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Topic Analysis")
                    st.info(f"**Topic #{topic['topic_id']}:** {topic['topic_name']}")
                    st.metric("Confidence", f"{topic['confidence']:.1%}")
                    if topic["keywords"]:
                        st.write("**Keywords:** " + ", ".join(topic["keywords"][:5]))

                with col2:
                    st.subheader("🔍 Fake News Detection")
                    if classification["prediction"] == "FAKE":
                        st.error("🚨 FAKE NEWS")
                    else:
                        st.success("✅ REAL NEWS")
                    st.metric("Confidence", f"{classification['confidence']:.1%}")

                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=["Real", "Fake"],
                                y=[
                                    classification["real_probability"],
                                    classification["fake_probability"],
                                ],
                                marker_color=["#10B981", "#EF4444"],
                            )
                        ]
                    )
                    fig.update_layout(yaxis_range=[0, 1], height=250)
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "📥 Download Results",
                    json.dumps(result, indent=2),
                    "analysis.json",
                    "application/json",
                )

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {e}")

st.divider()
st.caption("**Frontend:** Streamlit | **Backend:** FastAPI")
