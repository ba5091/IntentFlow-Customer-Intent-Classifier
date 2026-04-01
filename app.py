import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Intent Classifier Pro", page_icon="🤖", layout="wide")

# --- MODEL LOADING (With Caching for Performance) ---
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load("intent_model.pkl")
        vectorizer = joblib.load("tfidf.pkl")
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model_assets()

# Helper function for text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

# --- INITIALIZE SESSION STATE ---
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# --- SIDEBAR SECTION ---
st.sidebar.title("📊 Model Analytics")

def clear_text():
    st.session_state.query_input = ""

if st.sidebar.button("Clear All Inputs"):
    clear_text()
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.checkbox("Show Model Metrics"):
    st.sidebar.subheader("Model Evaluation")
    st.sidebar.metric(label="Overall Accuracy", value="93%")
    metrics_data = {
        "Intent": ["switch_account", "track_order", "track_refund"],
        "Precision": ["0.78", "0.83", "1.00"],
        "Recall": ["1.00", "0.83", "1.00"],
        "F1-Score": ["0.88", "0.83", "1.00"]
    }
    st.sidebar.table(pd.DataFrame(metrics_data))

st.sidebar.markdown("---")
st.sidebar.write("### 💡 Try these examples:")

if st.sidebar.button("📦 Track my package"):
    st.session_state.query_input = "Where is my order?"
if st.sidebar.button("💰 Refund status"):
    st.session_state.query_input = "I haven't received my refund yet"

# --- MAIN INTERFACE ---
st.title("📩 Customer Intent Classifier")
st.markdown("Automate ticket routing with NLP-powered intent detection.")

if model is None or vectorizer is None:
    st.error("🚨 Model files not found. Please run 'customer_intent.py' first to generate 'intent_model.pkl' and 'tfidf.pkl'.")
    st.stop()

tab1, tab2 = st.tabs(["Single Query", "Batch Processing"])

with tab1:
    user_input = st.text_input(
        "Enter customer message:", 
        value=st.session_state.query_input, 
        placeholder="e.g., I want to change my password..."
    )

    if st.button("Analyze Intent", type="primary"):
        if user_input:
            cleaned_input = clean_text(user_input)
            input_vector = vectorizer.transform([cleaned_input])
            
            # Prediction Logic
            probs = model.predict_proba(input_vector)[0]
            classes = model.classes_
            intent_probs = sorted(list(zip(classes, probs)), key=lambda x: x[1], reverse=True)[:3]
            
            top_intent, top_conf = intent_probs[0]

            # 1. Confidence Threshold Warning
            if top_conf < 0.60:
                st.warning(f"⚠️ **Low Confidence ({top_conf*100:.1f}%)**: This query may require human review.")
            else:
                st.success(f"✅ **Primary Intent: {top_intent}**")

            # 2. Results & Explainability
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write(f"**Confidence Score:** {top_conf*100:.2f}%")
                st.progress(float(top_conf))
                
                st.write("### Probability Breakdown")
                for intent, prob in intent_probs:
                    st.write(f"{intent}: {prob*100:.1f}%")
                    st.progress(float(prob))

            with col2:
                st.write("### Key Word Influence")
                feature_names = vectorizer.get_feature_names_out()
                
                # Handling multi-class vs binary coefficient indexing
                class_idx = list(classes).index(top_intent)
                if len(classes) > 2:
                    coef = model.coef_[class_idx]
                else:
                    coef = model.coef_[0] if class_idx == 1 else -model.coef_[0]
                
                words_in_input = cleaned_input.split()
                word_weights = []
                for word in set(words_in_input):
                    if word in feature_names:
                        idx = np.where(feature_names == word)[0][0]
                        weight = coef[idx]
                        word_weights.append((word, weight))
                
                if word_weights:
                    weights_df = pd.DataFrame(word_weights, columns=['Word', 'Impact']).sort_values(by='Impact', ascending=False)
                    st.bar_chart(weights_df.set_index('Word'))
                else:
                    st.info("No specific keywords strongly influenced this classification.")
        else:
            st.info("Please enter a message or select an example from the sidebar.")

with tab2:
    st.header("Batch File Processing")
    uploaded_file = st.file_uploader("Upload a CSV file for automated classification", type=["csv"])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(batch_df.head())
        
        column_to_process = st.selectbox("Select the column containing customer text:", batch_df.columns)
        
        if st.button("Run Batch Classification"):
            with st.spinner("Processing queries..."):
                batch_df['cleaned_text'] = batch_df[column_to_process].astype(str).apply(clean_text)
                batch_vectors = vectorizer.transform(batch_df['cleaned_text'])
                batch_df['predicted_intent'] = model.predict(batch_vectors)
                
                batch_probs = model.predict_proba(batch_vectors)
                batch_df['confidence'] = [max(p) for p in batch_probs]
                
                st.success("Classification Complete!")
                st.dataframe(batch_df[[column_to_process, 'predicted_intent', 'confidence']])
                
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="classified_customer_intents.csv",
                    mime="text/csv"
                )