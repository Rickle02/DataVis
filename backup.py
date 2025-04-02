import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.title()
    return df

DEFAULT_DATASET = "data/train_dataset.csv"
df = load_dataset(DEFAULT_DATASET)

# Precompute feature value counts
feature_counts = (
    df[["Iot", "Lte/5G", "Gbr", "Non-Gbr", "Ar/Vr/Gaming", "Healthcare", "Industry 4.0",
        "Iot Devices", "Public Safety", "Smart City & Home", "Smart Transportation", "Smartphone"]]
    .sum()
    .reset_index()
    .rename(columns={"index": "Feature", 0: "Count"})
)

available_line_features = [
    "Iot Devices",
    "Smart City & Home",
    "Smart Transportation",
    "Smartphone",
    "Public Safety",
    "Healthcare",
    "Industry 4.0",
    "Ar/Vr/Gaming"
]

# Init Dash app
app = dash.Dash(__name__)
app.title = "5G: Transforming Industries, Visualizing the Impact"


# Helper to filter data by category
def filter_data(category):
    if category == 'All':
        return df
    return df[df["Lte/5G Category"] == category]

# Layout
app.layout = html.Div([
    html.Label("Select Dataset:", style={"fontWeight": "bold", "margin": "10px"}),
    dcc.Dropdown(
        id='dataset-selector',
        options=[
            {"label": "Train Dataset", "value": "data/train_dataset.csv"},
            {"label": "Test Dataset", "value": "data/test_dataset.csv"}
        ],
        value="data/train_dataset.csv",
        clearable=False,
        style={"width": "50%", "margin": "0 auto 20px"}
    ),
    dcc.Store(id="stored-data"),  # 🔹 Added to store loaded dataset
    dcc.Tabs([
        dcc.Tab(label="Network Delay by Category", children=[
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': cat, 'value': cat} for cat in sorted(df["Lte/5G Category"].dropna().unique())] + [
                    {'label': 'All', 'value': 'All'}],
                value='All',
                clearable=False,
                placeholder="Select LTE/5G Category",
                style={'width': '50%', 'margin': '20px auto'}
            ),
            dcc.Graph(id="bar-chart")
        ]),
        dcc.Tab(label="CDF of Packet Delay", children=[
            html.Div(id="cdf-packet-delay")
        ]),
        dcc.Tab(label="Smartphone vs Others: Delay", children=[
            html.Div(id="smartphone-delay-boxplot-container")
        ]),
        dcc.Tab(label="Correlation of Network Features", children=[
            dcc.Graph(id="heatmap",
                      style={"height": "800px"},
                      figure=px.imshow(
                          df.drop(columns=["Slice Type"], errors='ignore')
                          .select_dtypes(include='number')
                          .corr(),
                          color_continuous_scale="RdBu",
                          title="Correlation Matrix of 5G Metrics (Excluding Slice Type)"
                      ))
        ]),
        dcc.Tab(label="Usage Over Time by Slice Type", children=[
            html.Div(id="slice-usage-container")
        ]),
        dcc.Tab(label="Slice Type vs Use Case Activation", children=[
            html.Div(id="slice-usecase-heatmap-container")
        ]),
        dcc.Tab(label="Slice Type Distribution", children=[
            html.Div(id="slice-type-container")
        ]),
        # dcc.Tab(label="Network Stability Over Time", children=[
        #     dcc.Graph(
        #         id="line-chart",
        #         figure=px.line(
        #             df.sort_values("Time"), x="Time",
        #             y=["Packet Delay", "Packet Loss Rate"],
        #             title="Time Series of Packet Delay and Loss Rate"
        #         )
        #     )
        # ]),
        dcc.Tab(label="GBR vs Non-GBR with other factors", children=[
            html.Div([
                dcc.Dropdown(
                    id="line-feature-selector",
                    options=[{"label": f, "value": f} for f in available_line_features],
                    multi=True,
                    placeholder="Select factors to compare",
                    maxHeight=150,
                    style={"width": "80%", "margin": "10px auto 20px"},
                    value=[]
                ),
                dcc.Graph(id="gbr-non-gbr-iot")
            ])
        ]),
        dcc.Tab(label="Feature Usage Scatterplot", children=[
            dcc.Dropdown(
                id='scatter-feature-dropdown',
                options=[
                            {'label': 'All', 'value': 'All'}
                        ] + [{'label': col, 'value': col} for col in [
                    "Iot", "Gbr", "Non-Gbr", "Ar/Vr/Gaming", "Healthcare",
                    "Industry 4.0", "Iot Devices", "Public Safety",
                    "Smart City & Home", "Smart Transportation", "Smartphone"
                ]],
                value="All",
                clearable=False,
                style={'width': '50%', 'margin': '20px auto'}
            ),
            dcc.Graph(
                id="scatter-feature-usage",
                style={"height": "700px"}  # You can adjust this value (e.g., 800px, 900px)
            )
        ])
    ])
])


# Callbacks

# Load dataset and store it
@app.callback(
    Output("stored-data", "data"),
    Input("dataset-selector", "value")
)
def update_dataset(path):
    df = load_dataset(path)
    return df.to_dict("records")


# Bar chart (delay by category)
@app.callback(
    Output("bar-chart", "figure"),
    Input("category-filter", "value"),
    Input("stored-data", "data")
)
def update_bar_chart(category, data):
    df = pd.DataFrame(data)
    df["Packet Loss Rate % (x10000)"] = df["Packet Loss Rate"] * 10000

    if category != "All":
        df = df[df["Lte/5G Category"] == category]

    # df["Packet Loss Rate %"] = df["Packet Loss Rate"] * 100

    avg_values = df.groupby("Lte/5G Category")[["Packet Delay", "Packet Loss Rate % (x10000)"]].mean().reset_index()
    # Compute original values before scaling
    original_loss = df.groupby("Lte/5G Category")["Packet Loss Rate"].mean().reset_index()
    original_loss.columns = ["Lte/5G Category", "Original Packet Loss Rate"]

    # Create scaled column
    df["Packet Loss Rate x10000"] = df["Packet Loss Rate"] * 10000

    # Combine for chart
    avg_values = df.groupby("Lte/5G Category")[["Packet Delay", "Packet Loss Rate x10000"]].mean().reset_index()
    melted = avg_values.melt(id_vars="Lte/5G Category", var_name="Metric", value_name="Value")

    # Replace names for chart legend
    melted["Metric"] = melted["Metric"].replace({
        "Packet Loss Rate x10000": "Packet Loss Rate (%)"
    })

    # Add hover info from original values
    melted = pd.merge(melted, original_loss, on="Lte/5G Category", how="left")
    melted["Custom Hover"] = melted.apply(lambda row:
                                          f"{row['Metric']}: {row['Value']:.2f}" if row["Metric"] == "Packet Delay"
                                          else f"{row['Metric']}: {row['Original Packet Loss Rate']:.5f}", axis=1)

    fig = px.bar(
        melted,
        x="Lte/5G Category",
        y="Value",
        color="Metric",
        barmode="stack",
        title="Average Packet Delay and Packet Loss Rate by LTE/5G Category",
        color_discrete_map={
            "Packet Delay": "#fc8d62",  # soft orange
            "Packet Loss Rate (%)": "#66c2a5"  # green
        }
    )

    # 👇 Override default hover info (this is the magic line)
    fig.update_traces(
        hovertemplate="%{customdata}"
    )

    # 👇 Inject your custom hover text as the trace's customdata
    fig.for_each_trace(
        lambda trace: trace.update(customdata=melted.loc[melted["Metric"] == trace.name, "Custom Hover"]))

    fig.update_layout(
        xaxis_title="LTE/5G Category",
        yaxis_title="Milliseconds / (Percentage x 10000)",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(t=60, b=100)
    )

    return fig


@app.callback(
    Output("cdf-packet-delay", "children"),
    Input("dataset-selector", "value"),
    Input("stored-data", "data")
)
def update_cdf_plot(dataset, data):
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np

    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip().str.title()

    if "Packet Delay" not in df.columns:
        return html.Div("Packet Delay column not found.", style={"textAlign": "center", "color": "red"})

    sorted_delay = np.sort(df["Packet Delay"])
    cdf = np.arange(len(sorted_delay)) / float(len(sorted_delay))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_delay,
        y=cdf,
        mode='lines',
        name='CDF',
        line=dict(color='#e6550d', width=2)
    ))

    fig.update_layout(
        title="CDF of Packet Delay",
        xaxis_title="Packet Delay (ms)",
        yaxis_title="Cumulative Probability",
        margin=dict(t=60, b=80, l=80, r=60)
    )

    return dcc.Graph(figure=fig)


# Slice Type vs Use Case Activation
@app.callback(
    Output("slice-usecase-heatmap-container", "children"),
    Input("dataset-selector", "value"),
    Input("stored-data", "data")
)
def update_slice_usecase_heatmap(dataset, data):
    import plotly.express as px
    import pandas as pd

    if dataset != "data/train_dataset.csv":
        return html.Div("This visualization is only available for the training dataset.",
                        style={"textAlign": "center", "marginTop": "20px", "color": "gray"})

    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip().str.title()

    # Mapping for Slice Type
    slice_type_map = {1: "eMBB", 2: "URLLC", 3: "mMTC"}
    df["Slice Type Label"] = df["Slice Type"].map(slice_type_map)

    use_cases = [
        "Healthcare", "Industry 4.0", "Iot Devices",
        "Public Safety", "Smart City & Home", "Smart Transportation"
    ]

    heat_df = df.groupby("Slice Type")[use_cases].sum().reset_index()
    heat_df = heat_df.melt(id_vars="Slice Type", var_name="Use Case", value_name="Count")

    fig = px.density_heatmap(
        heat_df,
        x="Use Case", y="Slice Type", z="Count",
        color_continuous_scale="Blues",
        title="Slice Type vs Use Case Activation (Count)",
    )

    fig.update_layout(
        title="Slice Type vs Use Case Activation<br>(Color: Avg Packet Delay, Text: Activation Count)",
        xaxis_title="Use Case",
        yaxis_title="Slice Type",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=["1  ", "2  ", "3  "],
            side='left',  # ✅ Back to left side
            tickfont=dict(size=14),
            automargin=True,
        ),
        margin=dict(t=60, b=80, l=60, r=60)  # ✅ Add more left space
    )

    return dcc.Graph(figure=fig)


# gbr
@app.callback(
    Output("gbr-non-gbr-iot", "figure"),
    Input("stored-data", "data"),
    Input("line-feature-selector", "value")
)
def update_gbr_non_gbr_iot(data, selected_features):
    df = pd.DataFrame(data)

    if not {"Lte/5G Category", "Gbr", "Non-Gbr"}.issubset(df.columns):
        return go.Figure()

    # Basic aggregation
    group_cols = ["Gbr", "Non-Gbr"] + (selected_features if selected_features else [])
    agg_df = df.groupby("Lte/5G Category")[group_cols].mean().reset_index()

    fig = go.Figure()

    # Bar chart: GBR & Non-GBR
    fig.add_bar(x=agg_df["Lte/5G Category"], y=agg_df["Gbr"], name="Avg GBR", marker_color="#3498db")
    fig.add_bar(x=agg_df["Lte/5G Category"], y=agg_df["Non-Gbr"], name="Avg Non-GBR", marker_color="#2ecc71")

    # Contrast color palette (hot and cool tones)
    contrast_colors = [
        "#e74c3c",  # red
        "#2980b9",  # blue
        "#f39c12",  # orange
        "#27ae60",  # green
        "#8e44ad",  # purple
        "#d35400",  # dark orange
        "#1abc9c",  # turquoise
        "#c0392b",  # dark red
    ]

    # Limit selected features to max 5
    # max_features = 5
    # if len(selected_features) > max_features:
    #     selected_features = selected_features[:max_features]

    # Add overlay line traces for selected features
    for i, feature in enumerate(selected_features):  # No limit now!
        fig.add_trace(go.Scatter(
            x=agg_df["Lte/5G Category"],
            y=agg_df[feature],
            name=f"Avg {feature}",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=contrast_colors[i % len(contrast_colors)], width=2)
        ))

    # Layout with dual axis
    fig.update_layout(
        title="GBR vs Non-GBR with Optional Feature Comparison",
        xaxis=dict(title="LTE/5G Category"),
        yaxis=dict(title="GBR / Non-GBR"),
        yaxis2=dict(title="Overlay Feature (e.g. IoT)", overlaying="y", side="right"),
        barmode="group",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(t=60, b=100)
    )

    return fig


# Slice
@app.callback(
    Output("slice-usage-container", "children"),
    Input("dataset-selector", "value"),
    Input("stored-data", "data")
)
def update_slice_usage_tab(dataset, data):
    if dataset != "data/train_dataset.csv":
        return html.Div("Slice Type distribution over time is only available for the training dataset.",
                        style={"textAlign": "center", "marginTop": "20px", "color": "gray"})

    df = pd.DataFrame(data)
    if "Slice Type" not in df.columns or "Time" not in df.columns:
        return html.Div("Slice Type or Time column not found in dataset.",
                        style={"textAlign": "center", "marginTop": "20px", "color": "red"})

    usage_time = df.groupby(['Time', 'Slice Type']).size().reset_index(name='Count')

    fig = px.area(
        usage_time,
        x="Time",
        y="Count",
        color="Slice Type",
        title="Usage Over Time by Slice Type",
        labels={"Count": "Connection Count", "Time": "Time"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(t=60, b=100)
    )

    return dcc.Graph(figure=fig)


# Smart
@app.callback(
    Output("smartphone-delay-boxplot-container", "children"),
    Input("dataset-selector", "value"),
    Input("stored-data", "data")
)
def update_smartphone_delay_plot(dataset, data):
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip().str.title()

    if "Smartphone" not in df.columns or "Packet Delay" not in df.columns:
        return html.Div("Required columns not found in dataset.",
                        style={"textAlign": "center", "marginTop": "20px", "color": "red"})

    df["Device Type"] = df["Smartphone"].map({1: "Smartphone", 0: "Other"})

    custom_colors = {
        "Smartphone": "#d95f0e",  # orange-brown
        "Other": "#fec44f"  # golden yellow
    }

    fig = px.box(
        df, x="Device Type", y="Packet Delay", color="Device Type",
        title="Packet Delay: Smartphone vs Other Devices",
        color_discrete_map=custom_colors
    )

    fig.update_layout(
        margin=dict(t=60, b=80),
        xaxis_title="Device Type",
        yaxis_title="Packet Delay (ms)",
        showlegend=False
    )

    return dcc.Graph(figure=fig)


# pie chart
@app.callback(
    Output("slice-type-container", "children"),
    Input("dataset-selector", "value"),
    Input("stored-data", "data")
)
def update_slice_type_tab(dataset, data):
    if dataset != "data/train_dataset.csv":
        return html.Div("Slice Type distribution is only available for the training dataset.",
                        style={"textAlign": "center", "marginTop": "20px", "color": "gray"})

    df = pd.DataFrame(data)
    if "Slice Type" not in df.columns:
        return html.Div("Slice Type column not found in dataset.",
                        style={"textAlign": "center", "marginTop": "20px", "color": "red"})

    slice_counts = df["Slice Type"].value_counts().sort_index().reset_index()
    slice_counts.columns = ["Slice Type", "Count"]

    # Optional: map numeric slice types to labels
    slice_counts["Slice Type"] = slice_counts["Slice Type"].map({
        1: "Type 1", 2: "Type 2", 3: "Type 3"
    })

    fig = px.pie(
        slice_counts,
        names="Slice Type",
        values="Count",
        title="Distribution of Slice Types (1, 2, 3)",
        color_discrete_sequence=px.colors.sequential.Blues[::-1]
    )

    fig.update_traces(
        textinfo='percent+label',  # Show label + %
        textposition='outside',
        pull=[0.05, 0.05, 0.05],  # Pull each slice slightly
        showlegend=True
    )

    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontal legend at bottom
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=100)
    )

    return dcc.Graph(figure=fig)


# Scatterplot feature usage
@app.callback(
    Output("scatter-feature-usage", "figure"),
    Input("scatter-feature-dropdown", "value"),
    Input("stored-data", "data")
)
def update_scatter_feature(selected, data):
    df = pd.DataFrame(data)
    if selected == "All":
        melt_df = df.melt(id_vars=["Lte/5G Category"],
                          value_vars=[
                              "Iot", "Gbr", "Non-Gbr", "Ar/Vr/Gaming", "Healthcare",
                              "Industry 4.0", "Iot Devices", "Public Safety",
                              "Smart City & Home", "Smart Transportation", "Smartphone"
                          ],
                          var_name="Feature", value_name="Value")
        melted_filtered = melt_df[melt_df["Value"] == 1]
        counts = melted_filtered.groupby(["Lte/5G Category", "Feature"]).size().reset_index(name="Count")
        fig = px.scatter(
            counts, x="Count", y="Lte/5G Category", color="Feature", size="Count",
            title="Feature Usage Count by LTE/5G Category (All Features)"
        )
    else:
        feature_counts = (
            df[df[selected] == 1]
            .groupby("Lte/5G Category")
            .size()
            .reset_index(name="Count")
        )
        fig = px.scatter(
            feature_counts,
            x="Count", y="Lte/5G Category",
            size="Count",
            color="Count",
            title=f"{selected} Usage Count by LTE/5G Category",
            color_continuous_scale="Turbo"
        )
    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
