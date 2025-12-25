import streamlit as st
from bertopic import BERTopic
import scipy.sparse as sp
import numpy as np
import torch

# Header
st.title("📰 Fake News Style Detector")
st.markdown(
    "Enter news text — the model will tell you if it's written in a "
    "fake news style and what topic it belongs to"
)


# ========== MONKEY PATCH for scipy.sparse.diags ==========
# Save original function
_original_diags = sp.diags


def patched_diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """Patch for scipy.sparse.diags that fixes BERTopic loading issue"""

    # If diagonals is a torch tensor
    if isinstance(diagonals, torch.Tensor):
        diagonals = diagonals.cpu().numpy().flatten()

    # If diagonals is a sparse matrix, extract diagonal
    elif sp.issparse(diagonals):
        diagonals = diagonals.diagonal()

    # Ensure it's a numpy array
    diagonals = np.asarray(diagonals).flatten()

    # If offsets is empty or None, set to 0
    if offsets is None or (
        isinstance(offsets, (list, tuple, np.ndarray)) and len(offsets) == 0
    ):
        offsets = 0

    # Call original function with corrected parameters
    return _original_diags(
        diagonals, offsets=offsets, shape=shape, format=format, dtype=dtype
    )


# Replace function BEFORE loading model
sp.diags = patched_diags
# ========== END MONKEY PATCH ==========


@st.cache_resource
def load_model():
    try:
        # Load model
        model = BERTopic.load("evstiars/bertopic-fake-news-detector")
        st.success("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        raise e


topic_model = load_model()

# Input field
user_text = st.text_area(
    "Enter news text:",
    height=200,
    placeholder="For example: Trump just said something shocking about Obama!"
)

if st.button("Analyze"):
    if user_text.strip():
        with st.spinner("Analyzing text..."):
            # Prediction
            topics, probs = topic_model.transform([user_text])
            topic_id = int(topics[0])
            prob = float(probs[0].max()) if probs.shape[1] > 1 else 1.0

            # Result
            if topic_id == -1:
                topic_name = "🤷 Outlier (didn't fit into any cluster)"
                keywords = []
            else:
                topic_name = f"Topic {topic_id}"
                try:
                    topic = topic_model.get_topic(topic_id)
                    keywords = [word for word, _ in topic[:10]]
                except Exception:
                    keywords = []

            st.success("Done!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Topic #", topic_id)
                st.write("**Topic Name:**", topic_name)
                st.write("**All topics**: ", topics)
            with col2:
                st.metric("Probability", f"{prob:.2%}")

            if keywords:
                st.write("**Top 10 Topic Keywords:**")
                st.write(", ".join(keywords))

            st.divider()
            st.write("**Entered Text:**")
            st.write(user_text)
    else:
        st.warning("Please enter text to analyze!")

# Footer
st.caption(
    "BERTopic model trained on 21k fake news articles from 2016–2018"
)
