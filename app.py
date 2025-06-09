import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Page config
st.set_page_config(page_title="Network Action Predictor", layout="wide")

# Load the trained model and label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset for preview and plotting
df = pd.read_csv("log2.csv")

# Add encoded action column if missing
if "Action_encoded" not in df.columns:
    df["Action_encoded"] = label_encoder.transform(df["Action"])

# Title
st.title("🔐 Network Traffic Action Predictor")
st.markdown("This app predicts whether a network traffic record should be **allowed** or **denied** based on traffic features.")

# -------------------------------
# Section 1: Data Preview
# -------------------------------
st.header("📄 Preview of Dataset")
st.dataframe(df.head())

# -------------------------------
# Section 2: Graphical Insights
# -------------------------------
st.header("📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Action Distribution")
    action_counts = df["Action"].value_counts()
    st.bar_chart(action_counts)

with col2:
    st.subheader("Bytes vs Packets")
    fig, ax = plt.subplots(figsize=(5, 4))  # Small size
    sns.scatterplot(data=df, x="Bytes", y="Packets", hue="Action", ax=ax)
    st.pyplot(fig)

# -------------------------------
# Section 3: User Input Form
# -------------------------------
st.header("🧾 Input Network Log for Prediction")

with st.form("input_form"):
    source_port = st.number_input("Source Port", value=12345)
    dest_port = st.number_input("Destination Port", value=443)
    nat_source_port = st.number_input("NAT Source Port", value=56789)
    nat_dest_port = st.number_input("NAT Destination Port", value=443)
    bytes_all = st.number_input("Bytes", value=1000)
    bytes_sent = st.number_input("Bytes Sent", value=500)
    bytes_received = st.number_input("Bytes Received", value=500)
    packets = st.number_input("Packets", value=10)
    elapsed_time = st.number_input("Elapsed Time (sec)", value=60)
    pkts_sent = st.number_input("Packets Sent", value=5)
    pkts_received = st.number_input("Packets Received", value=5)

    submit_button = st.form_submit_button("🔮 Predict")

# -------------------------------
# Section 4: Prediction Result
# -------------------------------
if submit_button:
    input_data = pd.DataFrame([[
        source_port, dest_port, nat_source_port, nat_dest_port,
        bytes_all, bytes_sent, bytes_received, packets,
        elapsed_time, pkts_sent, pkts_received
    ]], columns=[
        "Source Port", "Destination Port", "NAT Source Port", "NAT Destination Port",
        "Bytes", "Bytes Sent", "Bytes Received", "Packets",
        "Elapsed Time (sec)", "pkts_sent", "pkts_received"
    ])

    prediction = model.predict(input_data)[0]
    action_label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"✅ **Predicted Action: {action_label.upper()}**")

    # -------------------------------
    # Section 5: User Input Summary
    # -------------------------------
    st.subheader("📝 User Input Summary")
    st.table(input_data.T.rename(columns={0: "Value"}))

# -------------------------------
# Section 6: Model Evaluation
# -------------------------------
st.header("📉 Model Evaluation on Full Dataset")

# Feature columns
features = [
    "Source Port", "Destination Port", "NAT Source Port", "NAT Destination Port",
    "Bytes", "Bytes Sent", "Bytes Received", "Packets",
    "Elapsed Time (sec)", "pkts_sent", "pkts_received"
]

X = df[features]
y_true = df["Action_encoded"]
y_pred = model.predict(X)

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))  # Small figure
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

with col4:
    st.subheader("Classification Report")
    st.text(report)
