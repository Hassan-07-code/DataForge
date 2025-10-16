import streamlit as st
from modules.data_cleaning import DataCleaner

# --- Activity Logger Helper ---
def log_activity(section, message):
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = {}
    if section not in st.session_state.activity_log:
        st.session_state.activity_log[section] = []
    st.session_state.activity_log[section].append(message)

def render_cleaning(df_raw):
    st.header("🧹 Data Cleaning")
    st.markdown("Perform cleaning operations step by step with **full control**.")

    cleaner = DataCleaner(df_raw)

    st.markdown("### ⚙️ Choose Cleaning Operations")

    # --- Simple toggles (checkbox side by side) ---
    col1, col2 = st.columns(2)
    with col1:
        remove_dup = st.checkbox("Remove Duplicates")
    with col2:
        encode_cat = st.checkbox("Encode Categorical Variables")

    # --- Advanced operations (dropdown/expanders) ---
    strategy, cols_missing = None, None
    cols_encode, cols_scale, cols_remove = None, None, None

    with st.expander("🩹 Handle Missing Values"):
        strategy = st.selectbox("Select Strategy", ["mean", "median", "mode"])
        cols_missing = st.multiselect(
            "Select columns (leave empty for all)", df_raw.columns.tolist()
        )

    with st.expander("📏 Scale Numerical Features"):
        num_cols = df_raw.select_dtypes(include=["float64", "int64"]).columns.tolist()
        cols_scale = st.multiselect(
            "Select columns to scale (leave empty for all)", num_cols
        )

    with st.expander("❌ Remove Columns"):
        cols_remove = st.multiselect("Select columns to remove", df_raw.columns.tolist())

    # --- Apply Cleaning ---
    if st.button("✅ Apply Cleaning", type="primary", use_container_width=True):
        if remove_dup:
            cleaner.remove_duplicates()
        if strategy:
            cleaner.handle_missing(strategy=strategy, columns=cols_missing if cols_missing else None)
        if encode_cat:
            cleaner.encode_categoricals(columns=cols_encode if cols_encode else None)
        if cols_scale is not None and len(cols_scale) > 0:
            cleaner.scale_features(columns=cols_scale)
        if cols_remove is not None and len(cols_remove) > 0:
            cleaner.remove_columns(cols_remove)

        # Save cleaned dataset
        st.session_state.df_clean = cleaner.df

        # Log cleaning steps into activity_log
        if cleaner.log:
            for step in cleaner.log:
                log_activity("cleaning", step)

        # Show results
        st.markdown("### ✅ Cleaned Dataset Preview")
        st.dataframe(st.session_state.df_clean.head(20), use_container_width=True)

        # Cleaning log
        if cleaner.log:
            st.markdown("### 📝 Cleaning Steps Applied")
            for step in cleaner.log:
                st.info(step)

        # Next step
        st.success("➡️ Now proceed to **Visualization** from the sidebar.")
    else:
        st.info("👉 Select cleaning options and click **Apply Cleaning** to see results.")
