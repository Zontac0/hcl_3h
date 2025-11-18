import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

st.title("📊 Customer Churn Analytics Dashboard")

# Load CSV
st.sidebar.header("Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload your customer_churn_data.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Using your uploaded dataset from system…")
    df = pd.read_csv(r"C:\Users\LENOVO\Documents\vs_code\Notes\split\customer_churn_data.csv")

# -----------------------
# 🟦 SECTION 1 — DATA PREVIEW
# -----------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

st.subheader("🔍 Summary Statistics")
st.write(df.describe(include='all'))

# -----------------------
# 🟩 SECTION 2 — KPI METRICS
# -----------------------
st.subheader("📌 Key Performance Indicators (KPIs)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Users", len(df))

with col2:
    churn_rate = df["Churn"].mean() * 100
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}%")

with col3:
    avg_usage = df["MonthlyUsageHours"].mean()
    st.metric("Avg Monthly Usage Hours", f"{avg_usage:.2f}")

# -----------------------
# 🟨 SECTION 3 — VISUALIZATIONS
# -----------------------
st.subheader("📈 Visual Insights")
colA, colB = st.columns(2)

# Age distribution
with colA:
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], kde=True, ax=ax)
    st.pyplot(fig)

# Gender distribution
with colB:
    st.write("### Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Gender"], ax=ax)
    st.pyplot(fig)

colC, colD = st.columns(2)

# Subscription Type
with colC:
    st.write("### Subscription Type Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["SubscriptionType"], ax=ax)
    st.pyplot(fig)

# Complaints vs Churn Boxplot
with colD:
    st.write("### Complaints vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="Complaints", data=df, ax=ax)
    st.pyplot(fig)

# Scatter Plot
st.write("### Usage Hours vs Transactions (Colored by Churn)")
fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="MonthlyUsageHours",
    y="NumTransactions",
    hue="Churn",
    palette="coolwarm",
    ax=ax,
)
st.pyplot(fig)

# Heatmap
st.write("### 🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.success("Dashboard Rendered Successfully ✔️")
