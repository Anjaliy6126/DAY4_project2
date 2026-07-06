import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ------------------ Page Config ------------------
st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e3a8a,#312e81);
    color:white;
}
h1,h2,h3{color:white;}
[data-testid="stSidebar"]{
    background:#111827;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.title("📊 Customer Segmentation using K-Means")
st.write("Group customers based on **Age** and **Income** using **K-Means Clustering**.")

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Settings")
k = st.sidebar.slider("Number of Clusters", 2, 10, 3)

st.sidebar.markdown("### 🔗 GitHub")
st.sidebar.link_button(
    "View Project",
    "https://github.com/Anjaliy6126/DAY4_project2"
)

# ------------------ Dataset ------------------
df = pd.read_csv("income.csv")

st.subheader("📋 Dataset")
st.dataframe(df, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Customers", len(df))
c2.metric("Features", 2)

# ------------------ Original Plot ------------------
st.subheader("📈 Original Data")

fig, ax = plt.subplots()
ax.scatter(df["Age"], df["Income($)"], color="#38bdf8", s=70)
ax.set_xlabel("Age")
ax.set_ylabel("Income")
st.pyplot(fig)

# ------------------ Scaling ------------------
scaled = df.copy()

scaler = MinMaxScaler()
scaled["Age"] = scaler.fit_transform(scaled[["Age"]])
scaled["Income($)"] = scaler.fit_transform(scaled[["Income($)"]])

# ------------------ KMeans ------------------
km = KMeans(n_clusters=k, random_state=42)
scaled["Cluster"] = km.fit_predict(scaled[["Age", "Income($)"]])

# ------------------ Cluster Plot ------------------
st.subheader("🎯 Customer Segments")

colors = ["red","green","blue","orange","purple","pink","brown","cyan","gray","olive"]

fig, ax = plt.subplots(figsize=(7,5))

for i in range(k):
    d = scaled[scaled.Cluster == i]
    ax.scatter(d["Age"], d["Income($)"], s=80, color=colors[i], label=f"Cluster {i+1}")

ax.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    marker="*",
    s=250,
    color="yellow",
    label="Centroids"
)

ax.set_xlabel("Scaled Age")
ax.set_ylabel("Scaled Income")
ax.legend()

st.pyplot(fig)

# ------------------ Clustered Data ------------------
st.subheader("📊 Clustered Dataset")
st.dataframe(scaled, use_container_width=True)

# ------------------ Elbow Method ------------------
st.subheader("📉 Elbow Method")

sse = []

for i in range(1,11):
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(scaled[["Age","Income($)"]])
    sse.append(model.inertia_)

fig, ax = plt.subplots()

ax.plot(range(1,11), sse, marker="o", color="#38bdf8")
ax.set_xlabel("K")
ax.set_ylabel("SSE")
ax.grid(True)

st.pyplot(fig)

st.info("💡 The elbow point helps identify the optimal number of clusters.")

st.markdown("---")
st.markdown(
    "**Developed by Anjali | AI/ML Internship | Customer Segmentation Dashboard**"
)
