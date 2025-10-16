import streamlit as st
from modules.visualization import Visualizer

def render_visualization(df_clean):
    st.header("📊 Data Visualization")
    st.markdown("Explore your dataset with **interactive visualizations** powered by Plotly.")

    # ✅ Initialize activity log if not exists
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = {"visualizations": [], "models": [], "exports": []}

    vis = Visualizer(df_clean)

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Histogram",
            "Boxplot",
            "Scatter",
            "Line Chart",
            "Bar Chart",
            "Pie Chart",
            "Violin Plot",
            "KDE Density Plot",
            "Correlation Heatmap"
        ]
    )

    # --- Histogram ---
    if chart_type == "Histogram":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Histograms are great for understanding the **distribution of a variable**, detecting skewness, and spotting outliers.")
        col = st.selectbox("Select Column", df_clean.columns)
        bins = st.slider("Number of bins", 5, 100, 30)
        vis.plot_histogram(col, bins=bins)
        st.session_state.activity_log["visualizations"].append({
            "type": "Histogram", "column": col, "bins": bins
        })

    # --- Boxplot ---
    elif chart_type == "Boxplot":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Boxplots summarize the **spread of data, median, quartiles**, and highlight **outliers** effectively.")
        col = st.selectbox("Select Column", df_clean.columns)
        vis.plot_boxplot(col)
        st.session_state.activity_log["visualizations"].append({
            "type": "Boxplot", "column": col
        })

    # --- Scatter ---
    elif chart_type == "Scatter":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Scatter plots visualize **relationships between two variables**. Adding a trendline helps reveal correlation patterns.")
        x_col = st.selectbox("X-axis", df_clean.columns)
        y_col = st.selectbox("Y-axis", df_clean.columns)
        color_col = st.selectbox("Color Group (Optional)", [None] + df_clean.columns.tolist())
        add_trend = st.checkbox("Add Trendline")
        vis.plot_scatter(
            x_col,
            y_col,
            color=color_col if color_col != "None" else None,
            trendline=add_trend
        )
        st.session_state.activity_log["visualizations"].append({
            "type": "Scatter", "x": x_col, "y": y_col, "color": color_col, "trendline": add_trend
        })

    # --- Line Chart ---
    elif chart_type == "Line Chart":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Line charts are best for showing **trends over time** or continuous progression of variables.")
        x_col = st.selectbox("X-axis", df_clean.columns)
        y_col = st.selectbox("Y-axis", df_clean.columns)
        vis.plot_line(x_col, y_col)
        st.session_state.activity_log["visualizations"].append({
            "type": "Line Chart", "x": x_col, "y": y_col
        })

    # --- Bar Chart ---
    elif chart_type == "Bar Chart":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Bar charts compare **categories** easily and are great for analyzing discrete group differences.")
        x_col = st.selectbox("X-axis", df_clean.columns)
        y_col = st.selectbox("Y-axis", df_clean.columns)
        vis.plot_bar(x_col, y_col)
        st.session_state.activity_log["visualizations"].append({
            "type": "Bar Chart", "x": x_col, "y": y_col
        })

    # --- Pie Chart ---
    elif chart_type == "Pie Chart":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Pie charts show **proportions and percentages** of categories in a dataset.")
        col = st.selectbox("Column for Pie Chart", df_clean.columns)
        vis.plot_pie(col)
        st.session_state.activity_log["visualizations"].append({
            "type": "Pie Chart", "column": col
        })

    # --- Violin Plot ---
    elif chart_type == "Violin Plot":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Violin plots combine **boxplots and density plots**, showing distribution shape along with summary statistics.")
        col = st.selectbox("Select Column", df_clean.columns)
        vis.plot_violin(col)
        st.session_state.activity_log["visualizations"].append({
            "type": "Violin Plot", "column": col
        })

    # --- KDE Density Plot ---
    elif chart_type == "KDE Density Plot":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("KDE plots are useful for visualizing the **probability distribution** of continuous variables.")
        col = st.selectbox("Select Column", df_clean.columns)
        vis.plot_density(col)
        st.session_state.activity_log["visualizations"].append({
            "type": "KDE Density Plot", "column": col
        })

    # --- Correlation Heatmap ---
    elif chart_type == "Correlation Heatmap":
        st.markdown("### 📌 Chart Benefits & Usage")
        st.info("Correlation heatmaps identify **relationships between numerical features** and help in **feature selection** for modeling.")
        vis.plot_correlation()
        st.session_state.activity_log["visualizations"].append({
            "type": "Correlation Heatmap"
        })

    st.success("➡️ Next: Go to **Model Training** to build predictive models.")
