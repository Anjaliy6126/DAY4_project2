import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#0f172a,#1e3a8a,#312e81);
color:white;
}
.block-container{padding-top:2rem;}
.card{
background:rgba(255,255,255,0.08);
padding:18px;border-radius:15px;
border:1px solid rgba(255,255,255,.15);
}
h1,h2,h3{color:#f8fafc;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Segmentation using K-Means Clustering")
st.write("Group customers based on **Age** and **Income** using **K-Means Clustering**.")

st.markdown("### 🔗 GitHub Repository")
st.markdown("[![GitHub](https://img.shields.io/badge/View%20Project-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Anjaliy6126/DAY4_project2)")

k=st.sidebar.slider("Number of Clusters",2,10,3)

df=pd.read_csv("income.csv")
st.subheader("Dataset")
st.dataframe(df,use_container_width=True)

c1,c2=st.columns(2)
with c1:
    st.metric("Customers",len(df))
with c2:
    st.metric("Features",2)

fig,ax=plt.subplots(figsize=(6,4))
ax.scatter(df["Age"],df["Income($)"],color="cyan")
ax.set_title("Original Data")
ax.set_xlabel("Age")
ax.set_ylabel("Income")
st.pyplot(fig)

scaled=df.copy()
scaler=MinMaxScaler()
scaled["Age"]=scaler.fit_transform(scaled[["Age"]])
scaled["Income($)"]=scaler.fit_transform(scaled[["Income($)"]])

km=KMeans(n_clusters=k,random_state=42)
scaled["Cluster"]=km.fit_predict(scaled[["Age","Income($)"]])

colors=["#ff4b4b","#22c55e","#3b82f6","#f59e0b","#a855f7","#ec4899","#14b8a6","#64748b","#84cc16","#06b6d4"]

fig,ax=plt.subplots(figsize=(7,5))
for i in range(k):
    d=scaled[scaled.Cluster==i]
    ax.scatter(d["Age"],d["Income($)"],s=70,color=colors[i],label=f"Cluster {i+1}")
cent=km.cluster_centers_
ax.scatter(cent[:,0],cent[:,1],marker="*",s=300,color="yellow",label="Centroids")
ax.set_title("Customer Segments")
ax.set_xlabel("Scaled Age")
ax.set_ylabel("Scaled Income")
ax.legend()
st.pyplot(fig)

st.subheader("Clustered Dataset")
st.dataframe(scaled,use_container_width=True)

st.subheader("Customers in Each Cluster")
st.bar_chart(scaled["Cluster"].value_counts().sort_index())

sse=[]
for i in range(1,11):
    m=KMeans(n_clusters=i,random_state=42)
    m.fit(scaled[["Age","Income($)"]])
    sse.append(m.inertia_)
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(range(1,11),sse,marker="o",color="cyan")
ax.set_title("Elbow Method")
ax.set_xlabel("K")
ax.set_ylabel("SSE")
ax.grid(True)
st.pyplot(fig)

st.success("Project completed successfully!")

st.markdown("---")
st.markdown("""
### 💼 Project Summary
- Customer Segmentation using **K-Means Clustering**
- Feature Scaling with **MinMaxScaler**
- Optimal clusters using the **Elbow Method**

**Developed as part of my AI/ML Internship.**
""")
