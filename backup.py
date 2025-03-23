import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

# Title
st.title("5G Network Insights Dashboard")

# Sidebar for Dataset Selection
st.sidebar.header("Load Data")
train_path = "data/train_dataset.csv"
test_path = "data/test_dataset.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Format column names
train_df.columns = train_df.columns.str.strip().str.title()

st.subheader("Dataset Preview")
st.dataframe(train_df.head())

# 1. Bar Chart: Average Packet Delay by LTE/5G Category
if "Lte/5G Category" in train_df.columns and "Packet Delay" in train_df.columns:
    delay_bar = train_df.groupby("Lte/5G Category")["Packet Delay"].mean().reset_index()
    fig1 = px.bar(
        delay_bar, x="Lte/5G Category", y="Packet Delay",
        color="Lte/5G Category",
        title="Average Packet Delay by LTE/5G Category",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig1)

# 2. Heatmap: Correlation Heatmap
numeric_df = train_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
st.subheader("Correlation Heatmap of 5G Metrics")
fig2, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
st.pyplot(fig2)

# 3. Bubble Chart: Packet Loss vs Delay
if "Packet Loss Rate" in train_df.columns and "Packet Delay" in train_df.columns:
    fig3 = px.scatter(
        train_df,
        x="Packet Loss Rate", y="Packet Delay",
        size="Population" if "Population" in train_df.columns else None,
        color="Lte/5G Category",
        title="Packet Loss vs. Delay (Bubble Size = Population)",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig3)

# 4. Boxplot: IoT values by Category
if "Iot" in train_df.columns and "Lte/5G Category" in train_df.columns:
    fig4 = px.box(
        train_df, x="Lte/5G Category", y="Iot",
        title="IoT Distribution by LTE/5G Category",
        color="Lte/5G Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig4)

# 5. Histogram: Packet Loss Rate
if "Packet Loss Rate" in train_df.columns:
    fig5 = px.histogram(
        train_df, x="Packet Loss Rate",
        color="Lte/5G Category",
        title="Packet Loss Rate Distribution by Category",
        nbins=30,
        color_discrete_sequence=px.colors.sequential.Rainbow
    )
    st.plotly_chart(fig5)

# 6. Stacked Bar Chart: Traffic Types
if all(x in train_df.columns for x in ["Gbr", "Non-Gbr", "Iot", "Lte/5G Category"]):
    grouped = train_df.groupby("Lte/5G Category")[["Gbr", "Non-Gbr", "Iot"]].mean().reset_index()
    grouped = pd.melt(grouped, id_vars="Lte/5G Category", var_name="Traffic Type", value_name="Value")
    fig6 = px.bar(
        grouped, x="Lte/5G Category", y="Value",
        color="Traffic Type", barmode="stack",
        title="Average Traffic Types by LTE/5G Category",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig6)

# 7. Line Chart: Packet Delay and Loss Over Time
if "Time" in train_df.columns:
    sorted_df = train_df.sort_values("Time")
    fig7 = px.line(
        sorted_df, x="Time",
        y=["Packet Delay", "Packet Loss Rate"],
        title="Trends of Packet Delay and Loss Over Time",
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig7)

# 8. Multi-Panel Scatter Plot: Delay vs Key Metrics
metrics = ["Packet Loss Rate", "Iot", "Smart City & Home", "Healthcare"]
available_metrics = [m for m in metrics if m in train_df.columns]
fig8 = sp.make_subplots(rows=2, cols=2, subplot_titles=available_metrics)
row, col = 1, 1
for metric in available_metrics:
    fig8.add_trace(
        go.Scatter(
            x=train_df[metric], y=train_df["Packet Delay"],
            mode="markers", name=metric,
            marker=dict(color=np.random.rand(len(train_df)), colorscale="Viridis", showscale=False)
        ),
        row=row, col=col
    )
    col += 1
    if col > 2:
        row += 1
        col = 1
fig8.update_layout(title="Packet Delay vs Key Metrics", height=700, showlegend=False)
st.plotly_chart(fig8)

# End message
st.success("All charts rendered successfully!")
