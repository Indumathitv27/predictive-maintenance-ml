import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# ── API URL ────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Header ─────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance AI System")
st.markdown("Real-time turbofan engine health monitoring and RUL prediction using LightGBM.")
st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** LightGBM (Optuna tuned)
    **Dataset:** NASA CMAPS FD001
    **Val RMSE:** 10.59 cycles
    **Val R2:** 0.9407
    **Predicts:** Remaining Useful Life (RUL)
    """)
    st.divider()
    st.markdown("**Risk Levels:**")
    st.error("🔴 CRITICAL — RUL ≤ 20 cycles")
    st.warning("🟡 HIGH — RUL ≤ 50 cycles")
    st.info("🔵 MEDIUM — RUL ≤ 80 cycles")
    st.success("🟢 LOW — RUL > 80 cycles")
    st.divider()
    st.caption("For research use only.")

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Engine Prediction",
    "Batch Fleet Analysis",
    "Model Performance",
    "Snowflake History"
])

# ── TAB 1: Single Engine ───────────────────────────────────
with tab1:
    st.subheader("Single Engine Health Check")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Engine Info**")
        engine_id = st.number_input(
            "Engine ID", min_value=1, max_value=999, value=1
        )
        cycle = st.number_input(
            "Current Cycle", min_value=1, max_value=500, value=150
        )

        st.markdown("**Operational Settings**")
        op1 = st.number_input("Op Setting 1", value=-0.0007, format="%.4f")
        op2 = st.number_input("Op Setting 2", value=-0.0004, format="%.4f")
        op3 = st.number_input("Op Setting 3", value=100.0)

        st.markdown("**Key Sensor Readings**")
        s2 = st.number_input("Sensor 2 (Fan Inlet Temp)", value=641.82)
        s3 = st.number_input("Sensor 3 (LPC Outlet Temp)", value=1589.70)
        s4 = st.number_input("Sensor 4 (HPC Outlet Temp)", value=1400.60)
        s7 = st.number_input("Sensor 7 (Total HPC Outlet Pressure)", value=554.36)
        s11 = st.number_input("Sensor 11 (HPC Outlet Temp)", value=47.47)
        s12 = st.number_input("Sensor 12 (Fan Speed)", value=521.66)

    with col2:
        predict_btn = st.button(
            "🔍 Predict RUL",
            type="primary",
            use_container_width=True
        )

        if predict_btn:
            with st.spinner("Running inference..."):
                try:
                    payload = {
                        "engine_id": engine_id,
                        "cycle": cycle,
                        "op_setting_1": op1,
                        "op_setting_2": op2,
                        "op_setting_3": op3,
                        "sensor_2": s2,
                        "sensor_3": s3,
                        "sensor_4": s4,
                        "sensor_6": 21.61,
                        "sensor_7": s7,
                        "sensor_8": 2388.02,
                        "sensor_9": 9046.19,
                        "sensor_11": s11,
                        "sensor_12": s12,
                        "sensor_13": 2388.02,
                        "sensor_14": 8138.62,
                        "sensor_15": 8.4195,
                        "sensor_17": 392,
                        "sensor_20": 39.06,
                        "sensor_21": 23.419
                    }

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload
                    )
                    result = response.json()

                    rul = result["predicted_rul"]
                    risk = result["risk_level"]
                    rec = result["recommendation"]

                    # Risk color
                    if risk == "CRITICAL":
                        st.error(f"🔴 {risk} — RUL: {rul} cycles")
                    elif risk == "HIGH":
                        st.warning(f"🟡 {risk} — RUL: {rul} cycles")
                    elif risk == "MEDIUM":
                        st.info(f"🔵 {risk} — RUL: {rul} cycles")
                    else:
                        st.success(f"🟢 {risk} — RUL: {rul} cycles")

                    st.markdown(f"**Recommendation:** {rec}")
                    st.caption(f"Model: {result['model_version']}")

                    # RUL gauge chart
                    fig, ax = plt.subplots(
                        figsize=(6, 2),
                        facecolor="none"
                    )
                    max_rul = 130
                    color = (
                        "#ef4444" if risk == "CRITICAL"
                        else "#f59e0b" if risk == "HIGH"
                        else "#3b82f6" if risk == "MEDIUM"
                        else "#22c55e"
                    )
                    ax.barh(
                        ["RUL"], [rul],
                        color=color, height=0.4
                    )
                    ax.barh(
                        ["RUL"], [max_rul - rul],
                        left=[rul],
                        color="#e5e7eb", height=0.4
                    )
                    ax.set_xlim(0, max_rul)
                    ax.set_xlabel("Cycles remaining")
                    ax.set_title(
                        f"Engine {engine_id} — "
                        f"{rul} cycles remaining",
                        color="white"
                    )
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    fig.patch.set_alpha(0)
                    ax.set_facecolor("none")
                    st.pyplot(fig)
                    plt.close()

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure FastAPI is running.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Enter engine details and click Predict RUL.")

# ── TAB 2: Batch Fleet Analysis ────────────────────────────
with tab2:
    st.subheader("Fleet Health Overview")
    st.markdown("Simulate a fleet of engines at different cycle stages.")

    n_engines = st.slider("Number of engines to simulate", 5, 20, 10)

    if st.button("Run Fleet Analysis", type="primary"):
        with st.spinner("Analyzing fleet..."):
            np.random.seed(42)
            cycles = np.random.randint(50, 300, n_engines)

            readings = []
            for i, c in enumerate(cycles):
                degradation = c / 300.0
                readings.append({
                    "engine_id": i + 1,
                    "cycle": int(c),
                    "op_setting_1": -0.0007,
                    "op_setting_2": -0.0004,
                    "op_setting_3": 100.0,
                    "sensor_2": 641.82 + degradation * 3,
                    "sensor_3": 1589.70 + degradation * 15,
                    "sensor_4": 1400.60 + degradation * 20,
                    "sensor_6": 21.61,
                    "sensor_7": 554.36 + degradation * 5,
                    "sensor_8": 2388.02,
                    "sensor_9": 9046.19,
                    "sensor_11": 47.47 + degradation * 2,
                    "sensor_12": 521.66 + degradation * 3,
                    "sensor_13": 2388.02,
                    "sensor_14": 8138.62,
                    "sensor_15": 8.4195,
                    "sensor_17": 392,
                    "sensor_20": 39.06,
                    "sensor_21": 23.419
                })

            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    json=readings
                )
                data = response.json()
                predictions = data["predictions"]

                df = pd.DataFrame(predictions)
                df = df.sort_values("predicted_rul")

                # Color map
                def risk_color(risk):
                    return {
                        "CRITICAL": "#ef4444",
                        "HIGH": "#f59e0b",
                        "MEDIUM": "#3b82f6",
                        "LOW": "#22c55e"
                    }.get(risk, "#gray")

                colors = df["risk_level"].apply(risk_color)

                # Fleet chart
                fig, ax = plt.subplots(
                    figsize=(10, 4),
                    facecolor="none"
                )
                bars = ax.barh(
                    df["engine_id"].astype(str),
                    df["predicted_rul"],
                    color=colors
                )
                ax.axvline(x=20, color="#ef4444",
                           linestyle="--", alpha=0.7,
                           label="Critical threshold")
                ax.axvline(x=50, color="#f59e0b",
                           linestyle="--", alpha=0.7,
                           label="High threshold")
                ax.set_xlabel("Predicted RUL (cycles)",
                              color="white")
                ax.set_ylabel("Engine ID", color="white")
                ax.set_title(
                    "Fleet Engine Health Overview",
                    color="white"
                )
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.legend(facecolor="#1e2430",
                          labelcolor="white")
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                st.pyplot(fig)
                plt.close()

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                critical = len(df[df["risk_level"] == "CRITICAL"])
                high = len(df[df["risk_level"] == "HIGH"])
                medium = len(df[df["risk_level"] == "MEDIUM"])
                low = len(df[df["risk_level"] == "LOW"])

                c1.metric("🔴 Critical", critical)
                c2.metric("🟡 High", high)
                c3.metric("🔵 Medium", medium)
                c4.metric("🟢 Low", low)

                st.dataframe(
                    df[["engine_id", "cycle",
                        "predicted_rul", "risk_level",
                        "recommendation"]],
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Fleet analysis failed: {e}")

# ── TAB 3: Model Performance ───────────────────────────────
with tab3:
    st.subheader("Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "LightGBM")
    col2.metric("Val RMSE", "10.59 cycles")
    col3.metric("Val R2 Score", "0.9407")
    col4.metric("Optuna Trials", "50")

    st.divider()

    st.markdown("**Model Comparison**")
    comparison = pd.DataFrame({
        "Model": ["RandomForest", "XGBoost", "LightGBM", "LightGBM (Tuned)"],
        "Val RMSE": [11.34, 10.85, 10.71, 10.59],
        "Val R2": [0.932, 0.9377, 0.9393, 0.9407],
        "Status": ["Baseline", "Baseline", "Baseline", "Best"]
    })
    st.dataframe(comparison, use_container_width=True)

    st.divider()
    st.markdown("**ML Pipeline**")
    st.markdown("""
    1. **Data Ingestion** — NASA CMAPS FD001 dataset (20,631 training cycles)
    2. **Feature Engineering** — Rolling mean, rolling std, lag features, cycle normalization (48 features)
    3. **Model Training** — RandomForest, XGBoost, LightGBM with 5-fold cross-validation
    4. **Hyperparameter Tuning** — Optuna Bayesian optimization (50 trials)
    5. **MLflow Tracking** — All experiments logged with params, metrics, and model artifacts
    6. **Deployment** — FastAPI REST endpoint with risk classification
    """)

    st.divider()
    st.markdown("**AWS / GCP Production Mapping**")
    mapping = pd.DataFrame({
        "Local Component": [
            "LightGBM model",
            "FastAPI endpoint",
            "MLflow tracking",
            "Streamlit dashboard",
            "Feature pipeline",
            "Docker container"
        ],
        "AWS Equivalent": [
            "SageMaker endpoint",
            "API Gateway + Lambda",
            "SageMaker Experiments",
            "CloudFront + React",
            "AWS Glue / EMR",
            "ECR + ECS Fargate"
        ],
        "GCP Equivalent": [
            "Vertex AI endpoint",
            "Cloud Run + Cloud Functions",
            "Vertex AI Experiments",
            "Firebase Hosting",
            "Dataflow / Dataproc",
            "Artifact Registry + GKE"
        ]
    })
    st.dataframe(mapping, use_container_width=True)

# ── TAB 4: Snowflake History ───────────────────────────────
with tab4:
    st.subheader("Prediction History — Snowflake")
    st.markdown("Live prediction audit trail stored in Snowflake.")

    col1, col2 = st.columns([1, 3])

    with col1:
        limit = st.slider("Records to fetch", 10, 100, 25)
        refresh = st.button("Refresh from Snowflake",
                           type="primary",
                           use_container_width=True)

    if refresh:
        with st.spinner("Fetching from Snowflake..."):
            try:
                sys.path.append(
                    os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__)
                    ))
                )
                from api.snowflake_logger import (
                    get_prediction_history,
                    get_risk_summary
                )

                history = get_prediction_history(limit)
                summary = get_risk_summary()

                if history:
                    # Risk summary metrics
                    st.markdown("**Risk Distribution — All Time**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "🔴 Critical",
                        summary.get("CRITICAL", 0)
                    )
                    c2.metric(
                        "🟡 High",
                        summary.get("HIGH", 0)
                    )
                    c3.metric(
                        "🔵 Medium",
                        summary.get("MEDIUM", 0)
                    )
                    c4.metric(
                        "🟢 Low",
                        summary.get("LOW", 0)
                    )

                    st.divider()

                    # History table
                    st.markdown(
                        f"**Last {len(history)} predictions**"
                    )
                    df = pd.DataFrame(history)
                    df["predicted_at"] = pd.to_datetime(
                        df["predicted_at"]
                    ).dt.strftime("%Y-%m-%d %H:%M:%S")

                    st.dataframe(
                        df[[
                            "engine_id", "cycle",
                            "predicted_rul", "risk_level",
                            "recommendation", "predicted_at"
                        ]],
                        use_container_width=True
                    )

                    # RUL trend chart
                    st.divider()
                    st.markdown("**RUL Trend — Recent Predictions**")
                    fig, ax = plt.subplots(
                        figsize=(10, 3),
                        facecolor="none"
                    )
                    colors = df["risk_level"].map({
                        "CRITICAL": "#ef4444",
                        "HIGH": "#f59e0b",
                        "MEDIUM": "#3b82f6",
                        "LOW": "#22c55e"
                    }).fillna("#gray")

                    ax.scatter(
                        range(len(df)),
                        df["predicted_rul"],
                        c=colors,
                        s=80,
                        zorder=3
                    )
                    ax.plot(
                        range(len(df)),
                        df["predicted_rul"],
                        color="#4d5969",
                        linewidth=1,
                        zorder=2
                    )
                    ax.axhline(
                        y=20, color="#ef4444",
                        linestyle="--", alpha=0.6,
                        label="Critical threshold"
                    )
                    ax.axhline(
                        y=50, color="#f59e0b",
                        linestyle="--", alpha=0.6,
                        label="High threshold"
                    )
                    ax.set_xlabel(
                        "Prediction index", color="white"
                    )
                    ax.set_ylabel(
                        "Predicted RUL", color="white"
                    )
                    ax.set_title(
                        "RUL Predictions Over Time",
                        color="white"
                    )
                    ax.tick_params(colors="white")
                    ax.legend(
                        facecolor="#1e2430",
                        labelcolor="white"
                    )
                    fig.patch.set_alpha(0)
                    ax.set_facecolor("none")
                    st.pyplot(fig)
                    plt.close()

                else:
                    st.info(
                        "No predictions yet. Make some predictions "
                        "in Tab 1 or Tab 2 first."
                    )

            except Exception as e:
                st.error(f"Snowflake fetch failed: {e}")
    else:
        st.info("Click Refresh to load prediction history from Snowflake.")