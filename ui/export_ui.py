import streamlit as st
import os
from modules.report_export import Exporter

def render_export(reports_dir, models_dir):
    st.header("📤 Export & Reports")
    st.markdown("Download all outputs as reports, cleaned datasets, trained models, and activity logs.")

    exporter = Exporter(reports_dir, models_dir)

    if st.button("Generate & Download Report"):
        with st.spinner("Preparing your report package..."):
            zip_path = exporter.create_full_report(
                st.session_state.df_clean,
                st.session_state.activity_log,
                st.session_state.metrics
            )

        # Show contents preview
        st.subheader("📦 Your ZIP will contain:")

        base_files = [
            "`cleaned_dataset.csv`",
            "`activity_log.json`",
            "`report.pdf`"
        ]

        # Add models dynamically
        model_files = []
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                model_files.append(f"`models/{f}`")

        # Render list
        for bf in base_files:
            st.markdown(f"- {bf}")
        if model_files:
            for mf in model_files:
                st.markdown(f"- {mf}")

        # Download button
        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Download DataForge Report", f, file_name="dataforge_report.zip")

        st.success("✅ Report generated successfully!")

        # Log export activity
        if "exports" not in st.session_state.activity_log:
            st.session_state.activity_log["exports"] = []

        st.session_state.activity_log["exports"].append({
            "file": "dataforge_report.zip",
            "timestamp": str(st.session_state.get("timestamp", "now")),
            "status": "success"
        })
