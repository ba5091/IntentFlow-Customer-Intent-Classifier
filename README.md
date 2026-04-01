🤖 Customer Intent Classifier:A machine learning-powered web application designed to automate the classification of customer service inquiries. This project utilizes Natural Language Processing (NLP) to identify the "intent" behind a customer's message, helping support teams prioritize and route tickets efficiently.🌟 Key FeaturesReal-time Inference: Enter any customer query and receive an instant intent classification.Confidence Scoring: Displays the mathematical certainty of the model's top choice using probability estimates.Top-3 Probability Analysis: Shows the three most likely intents to provide transparency in ambiguous cases.Interactive Examples: Sidebar buttons allow users to test the model with pre-set industry-standard queries.Model Performance Dashboard: A toggleable section showing real-time metrics including accuracy and precision.🛠️ Tech StackFrontend: StreamlitMachine Learning: Scikit-learn (Logistic Regression)NLP Techniques: TF-IDF Vectorization, Text Standardization (Regex-based cleaning)Data Handling: PandasModel Persistence: Joblib (for .pkl serialization)📂 Project StructurePlaintextCustomer_Intent_App/
├── app.py                # Main Streamlit application code
├── customer_intent.py    # Model training and evaluation script
├── intent_model.pkl      # Serialized Logistic Regression model
├── tfidf.pkl             # Serialized TF-IDF Vectorizer
├── requirements.txt      # List of Python dependencies
├── dataset.csv           # Customer service utterance dataset
└── README.md             # Project documentation
🚀 Getting Started1. Set Up Virtual EnvironmentBashpython -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
2. Install DependenciesBashpip install -r requirements.txt
3. Run the ApplicationBashstreamlit run app.py
📊 Model EvaluationThe model was trained using a stratified split and achieved an Overall Accuracy of 93%.IntentPrecisionRecallF1-Scoreswitch_account0.781.000.88track_order0.830.830.83track_refund1.001.001.00