import streamlit as st
import pandas as pd

def render_home():
    st.title("🛠️ DataForge - Auto ML Area")
    st.subheader("Upload your dataset to get started")

    st.markdown("### 📥 Upload CSV or Excel file")
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV, Excel"
    )

    # Show user guidance
    with st.expander("ℹ️ How to use this tool?"):
        st.markdown("""
        - **Step 1:** Upload your dataset (CSV or Excel).
        - **Step 2:** Preview appears instantly below.
        - **Step 3:** Navigate to **Data Cleaning** from sidebar.
        - **Step 4:** Follow step-by-step workflow till Export.
        """)

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Save in session
            st.session_state.df_raw = df

            # Success message
            st.success(f"✅ File `{uploaded_file.name}` uploaded successfully!")

            # Dataset preview
            st.markdown("### 👀 Dataset Preview")
            # Dataset shape
            st.info(f"📊 Dataset Shape → Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.dataframe(df.head(20), use_container_width=True)

            st.info("➡️ Next: Go to **Data Cleaning** in the sidebar to start cleaning your data.")

        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
