import os
import pickle
import streamlit as st
import pandas as pd
from datetime import datetime
from modules.model_training import ModelTrainer

# init / ensure activity_log
def _init_activity_log():
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = {
            "cleaning": [],
            "visualizations": [],
            "model_training": [],
            "exports": []
        }
    else:
        for k in ["cleaning", "visualizations", "model_training", "exports"]:
            if k not in st.session_state.activity_log:
                st.session_state.activity_log[k] = []

def log_model_activity(entry: dict):
    _init_activity_log()
    st.session_state.activity_log["model_training"].append(entry)


def _persist_demo_in_session(model_obj, X_test, y_test, task, model_name):
    """
    Save demo-trained objects into session_state so they persist across Streamlit reruns.
    """
    st.session_state.demo = {
        "model": model_obj,
        "X_test": X_test,
        "y_test": y_test,
        "task": task,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat()
    }


def render_model_training(df_clean, models_dir):
    _init_activity_log()
    st.header("🤖 Model Training")
    st.markdown("Train ML models and compare performance. (Demo ➡️ Test ➡️ Final)")

    # target column selection
    target_col = st.selectbox("🎯 Select target column", df_clean.columns)
    if not target_col:
        st.warning("Please select a target column.")
        return

    # create a fresh trainer instance for helper methods; trainer instance state is not used across reruns
    trainer = ModelTrainer(df_clean, target_col, models_dir)
    suggested, reason = trainer.suggest_model()
    st.markdown(f"### 💡 Suggested: **{suggested}**")
    st.info(reason)

    # build available models list depending on detected problem
    task = trainer.detect_problem_type()
    if task == "classification":
        available_models = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Naive Bayes",
            "Gradient Boosting"
        ]
        # add XGBoost if available in trainer map
        try:
            if "XGBoost" in trainer._get_model_map("classification"):
                available_models.insert(3, "XGBoost")
        except Exception:
            pass
    else:
        available_models = [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "SVR",
            "Gradient Boosting Regressor"
        ]
        try:
            if "XGBoost Regressor" in trainer._get_model_map("regression"):
                available_models.insert(3, "XGBoost Regressor")
        except Exception:
            pass

    # default selection index prefers suggested model when present
    default_idx = available_models.index(suggested) if suggested in available_models else 0
    selected_model = st.selectbox("⚙️ Choose model", available_models, index=default_idx)

    st.markdown("---")

    # ---------------- Demo Training ----------------
    if st.button("🚀 Run Demo Training (80% train / 20% test)"):
        with st.spinner("Running demo training (this uses an 80/20 split)..."):
            try:
                model_obj, train_metrics, n_train, n_test = trainer.train_demo_model(selected_model, test_size=0.2)
            except Exception as e:
                st.error(f"Demo training failed: {e}")
                return

        # persist demo model + test split to session_state so Test works across reruns
        _persist_demo_in_session(model_obj, trainer.X_test, trainer.y_test, trainer.task, selected_model)

        st.success("✅ Demo training finished")
        st.markdown(f"**Train samples:** {n_train} | **Test samples (reserved):** {n_test}")
        st.markdown("### 📊 Training metrics (on 80% training partition)")
        st.dataframe(pd.DataFrame([train_metrics]), use_container_width=True)

        # log demo action
        log_model_activity({
            "timestamp": datetime.now().isoformat(),
            "stage": "demo-train",
            "target": target_col,
            "chosen_model": selected_model,
            "train_samples": int(n_train),
            "test_samples": int(n_test),
            "metrics": train_metrics
        })

    st.markdown("---")

    # ---------------- Test Model (evaluate on reserved 20%) ----------------
    if st.button("🧪 Test Model (evaluate on reserved 20%)"):
        # check demo presence in session_state
        demo = st.session_state.get("demo", None)
        if demo is None:
            st.error("No demo model or test set found. Please run Demo Training first.")
        else:
            # Use trainer's evaluation helpers (stateless) to get metrics on stored test split
            try:
                model_obj = demo["model"]
                X_test = demo["X_test"]
                y_test = demo["y_test"]
                task_demo = demo["task"]

                if X_test is None or y_test is None:
                    st.error("No test split available. Demo training may have failed earlier.")
                else:
                    # Evaluate using trainer helper functions
                    if task_demo == "classification":
                        test_metrics = trainer._evaluate_classification(model_obj, X_test, y_test)
                    else:
                        test_metrics = trainer._evaluate_regression(model_obj, X_test, y_test)

                    st.success("📊 Test evaluation completed")
                    st.markdown("### 🔎 Test metrics (on reserved 20% partition)")
                    st.dataframe(pd.DataFrame([test_metrics]), use_container_width=True)

                    # log demo-test
                    log_model_activity({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "demo-test",
                        "target": target_col,
                        "chosen_model": demo.get("model_name", selected_model),
                        "metrics": test_metrics
                    })

                    # Offer to save the demo-trained model
                    st.markdown("#### 💾 Save demo-trained model?")
                    save_demo = st.button("Save Demo Model (.pickle)")
                    if save_demo:
                        try:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe_name = (demo.get("model_name") or selected_model).replace(" ", "_")
                            model_filename = f"{safe_name}_demo_{ts}.pickle"
                            model_path = os.path.join(models_dir, model_filename)
                            with open(model_path, "wb") as f:
                                pickle.dump(model_obj, f)

                            # update session_state trained_models & metrics for UI/export
                            if "trained_models" not in st.session_state:
                                st.session_state.trained_models = {}
                            if "metrics" not in st.session_state:
                                st.session_state.metrics = {}

                            st.session_state.trained_models[safe_name + "_demo"] = model_path
                            st.session_state.metrics[safe_name + "_demo"] = test_metrics

                            st.success(f"Demo model saved: `{model_path}`")

                            # log export/save
                            log_model_activity({
                                "timestamp": datetime.now().isoformat(),
                                "stage": "demo-save",
                                "target": target_col,
                                "chosen_model": demo.get("model_name", selected_model),
                                "model_path": model_path,
                                "metrics": test_metrics
                            })
                        except Exception as e:
                            st.error(f"Failed to save demo model: {e}")
            except Exception as e:
                st.error(f"Testing failed: {e}")

    st.markdown("---")

    # ---------------- Final Training (train on full cleaned dataset & save) ----------------
    if st.button("🏁 Train Final Model (100% and save)"):
        with st.spinner("Training final model on full cleaned dataset..."):
            try:
                model_path, final_metrics = trainer.train_final_model(selected_model, save=True)
            except Exception as e:
                st.error(f"Final training failed: {e}")
                return

        st.success(f"🎉 Final model trained and saved: `{model_path}`")
        st.markdown("### 📊 Final model metrics (evaluated on full dataset)")
        st.dataframe(pd.DataFrame([final_metrics]), use_container_width=True)

        # update session-level storage for exports & UI
        if "trained_models" not in st.session_state:
            st.session_state.trained_models = {}
        if "metrics" not in st.session_state:
            st.session_state.metrics = {}

        name_key = selected_model.replace(" ", "_")
        st.session_state.trained_models[name_key] = model_path
        st.session_state.metrics[name_key] = final_metrics

        # log final training
        log_model_activity({
            "timestamp": datetime.now().isoformat(),
            "stage": "final",
            "target": target_col,
            "chosen_model": selected_model,
            "model_path": model_path,
            "metrics": final_metrics
        })

        st.info("➡️ After this, go to Export to generate the full ZIP report containing models, metrics and report.")
