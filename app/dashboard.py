import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.assistant.rag_engine import DecisionAssistant
from src.causal.diagnostics import CausalDiagnostics
from src.data.loader import UpliftDataLoader
from src.evaluation.evaluator import UpliftEvaluator
from src.features.preprocessor import CausalPreprocessor
from src.policy.optimizer import PolicyOptimizer
from src.policy.segmenter import CustomerSegmenter

# Streamlit Page Setup
st.set_page_config(
    page_title="DeltaCausal Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = Path("artifacts")


def load_available_experiments():
    """List available experiment runs."""
    if not ARTIFACTS_DIR.exists():
        return []
    runs = sorted([d.name for d in ARTIFACTS_DIR.iterdir() if d.is_dir() and (d.name.startswith("exp_") or d.name == "smoke_test_exp")])
    return runs


def load_experiment_data(exp_id: str):
    """Load JSON/CSV artifacts for selected experiment."""
    exp_dir = ARTIFACTS_DIR / exp_id
    data = {}

    eval_p = exp_dir / "evaluation_report.json"
    if eval_p.exists():
        with open(eval_p, "r", encoding="utf-8") as f:
            data["evaluation"] = json.load(f)

    diag_p = exp_dir / "causal_diagnostics.json"
    if diag_p.exists():
        with open(diag_p, "r", encoding="utf-8") as f:
            data["diagnostics"] = json.load(f)

    seg_p = exp_dir / "customer_segments.json"
    if seg_p.exists():
        with open(seg_p, "r", encoding="utf-8") as f:
            data["segments"] = json.load(f)

    pol_p = exp_dir / "policy_comparison.csv"
    if pol_p.exists():
        data["policy_df"] = pd.read_csv(pol_p)

    val_comp_p = exp_dir / "val_model_comparison.csv"
    if val_comp_p.exists():
        data["val_comparison_df"] = pd.read_csv(val_comp_p)

    test_comp_p = exp_dir / "test_model_comparison.csv"
    if test_comp_p.exists():
        data["test_comparison_df"] = pd.read_csv(test_comp_p)

    sens_p = exp_dir / "sensitivity_analysis.csv"
    if sens_p.exists():
        data["sensitivity_df"] = pd.read_csv(sens_p)

    return data


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎯 DeltaCausal")
st.sidebar.markdown("**Counterfactual Business Decision Platform**")

exp_runs = load_available_experiments()
selected_exp = st.sidebar.selectbox("Active Experiment", exp_runs if exp_runs else ["No Runs Found"])

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "1. Executive Overview",
        "2. Data & Experiment Diagnostics",
        "3. Model Comparison & Benchmark",
        "4. Customer Targeting & Segmentation",
        "5. Business Simulator & Policy",
        "6. Causal Decision Assistant",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Causal Principle**: Conventional propensity models target *likely buyers*. "
    "Counterfactual uplift models target *persuadable customers*—maximizing incremental profit."
)

exp_data = load_experiment_data(selected_exp) if exp_runs and selected_exp != "No Runs Found" else {}


# --- PAGE 1: EXECUTIVE OVERVIEW ---
if nav_selection == "1. Executive Overview":
    st.title("📊 Executive Overview: Causal Campaign Intelligence")
    st.markdown("High-level summary of estimated treatment effects, incremental campaign performance, and targeting ROI.")

    if not exp_data:
        st.warning("⚠️ No experiment data found. Please run the training pipeline first: `python scripts/run_pipeline.py`")
    else:
        eval_info = exp_data.get("evaluation", {})
        policy_df = exp_data.get("policy_df", pd.DataFrame())
        best_model = eval_info.get("best_model_name", "x_learner")

        # Top Metric Cards
        uplift_policy_row = policy_df[policy_df["policy_name"] == "Counterfactual Uplift Targeting"]
        random_policy_row = policy_df[policy_df["policy_name"] == "Random Targeting"]
        prop_policy_row = policy_df[policy_df["policy_name"] == "Response Propensity Targeting"]

        col1, col2, col3, col4 = st.columns(4)
        if not uplift_policy_row.empty:
            inc_profit = float(uplift_policy_row["expected_incremental_profit"].values[0])
            roi = float(uplift_policy_row["roi_percentage"].values[0])
            targeted = int(uplift_policy_row["customers_targeted"].values[0])
            inc_conv = float(uplift_policy_row["expected_incremental_conversions"].values[0])

            col1.metric("Incremental Profit", f"${inc_profit:,.2f}", delta=f"{roi:.1f}% ROI")
            col2.metric("Incremental Conversions", f"{inc_conv:,.1f}", delta="Lift over baseline")
            col3.metric("Targeted Customers", f"{targeted:,}", f"{uplift_policy_row['budget_spent'].values[0]:,.0f} spent")
            col4.metric("Selected Model", best_model.replace('_', ' ').title(), f"Metric: {eval_info.get('primary_metric')}")

        st.markdown("---")

        # Comparison Chart: Policy Net Value
        if not policy_df.empty:
            st.subheader("Campaign Financial Return by Targeting Strategy")
            fig = px.bar(
                policy_df,
                x="policy_name",
                y="expected_incremental_profit",
                color="policy_name",
                text="expected_incremental_profit",
                labels={"expected_incremental_profit": "Expected Incremental Profit ($)", "policy_name": "Targeting Policy"},
                title="Profit Comparison Under Identical Campaign Budget",
            )
            fig.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
            fig.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown(
            "### Critical Causal Caveat\n"
            "> **Estimated effects are Conditional Average Treatment Effects $\\tau(x) = E[Y(1) - Y(0) | X=x]$, "
            "not deterministic individual causal facts.** Unobserved customer attributes or external marketing channels "
            "can influence individual responsiveness. Targeting policies should maintain a randomized holdout group."
        )


# --- PAGE 2: DIAGNOSTICS ---
elif nav_selection == "2. Data & Experiment Diagnostics":
    st.title("🔬 Data Validation & Experimental Design Diagnostics")
    st.markdown("Inspect covariate balance, Sample Ratio Mismatch (SRM), and propensity score common support.")

    diag = exp_data.get("diagnostics", {})
    if not diag:
        st.info("No diagnostics file available for selected run.")
    else:
        col1, col2 = st.columns(2)
        srm = diag.get("srm_test", {})
        overlap = diag.get("propensity_overlap", {})

        with col1:
            st.subheader("Sample Ratio Mismatch (SRM) Test")
            srm_pass = not srm.get("is_srm_detected", False)
            status_color = "green" if srm_pass else "red"
            st.markdown(f"**Status**: :{status_color}[{'PASSED' if srm_pass else 'FAILED / FLAGGED'}]")
            st.markdown(f"- **Treated Count**: {srm.get('observed_treated', 0):,}")
            st.markdown(f"- **Control Count**: {srm.get('observed_control', 0):,}")
            st.markdown(f"- **Chi-Square Stat**: {srm.get('chi2_statistic', 0.0)}")
            st.markdown(f"- **P-Value**: {srm.get('p_value', 1.0):.6f}")
            st.caption(srm.get("interpretation", ""))

        with col2:
            st.subheader("Propensity Score Overlap / Positivity")
            overlap_pass = overlap.get("is_overlap_adequate", True)
            st.markdown(f"**Common Support**: `[{overlap.get('common_support_min', 0):.3f}, {overlap.get('common_support_max', 1):.3f}]`")
            st.markdown(f"- **Extreme Positivity Violations**: {overlap.get('violations_count', 0)} ({overlap.get('violations_pct', 0):.2%})")
            st.caption(overlap.get("summary", ""))

        st.markdown("---")
        st.subheader("Covariate Balance: Standardized Mean Differences (SMD)")
        st.markdown("Values with $|SMD| < 0.10$ indicate adequate balance between treatment and control.")

        cov_list = diag.get("covariate_balance", [])
        if cov_list:
            df_cov = pd.DataFrame(cov_list)
            fig_smd = px.bar(
                df_cov,
                x="feature",
                y="smd",
                color="is_imbalanced",
                color_discrete_map={False: "#2ca02c", True: "#d62728"},
                title="Standardized Mean Difference (SMD) by Pre-Treatment Feature",
                labels={"smd": "SMD", "feature": "Covariate", "is_imbalanced": "Imbalance Flag (|SMD| > 0.10)"},
            )
            fig_smd.add_hline(y=0.10, line_dash="dash", line_color="red")
            fig_smd.add_hline(y=-0.10, line_dash="dash", line_color="red")
            fig_smd.update_layout(template="plotly_white")
            st.plotly_chart(fig_smd, use_container_width=True)

            st.dataframe(df_cov[["feature", "mean_treated", "mean_control", "smd", "ks_p_value", "is_imbalanced"]], use_container_width=True)


# --- PAGE 3: MODEL COMPARISON ---
elif nav_selection == "3. Model Comparison & Benchmark":
    st.title("Model Comparison & Uplift Benchmark")
    st.markdown("Compare meta-learners against standard baselines across Qini curves, AUUC, and CATE estimation accuracy.")

    val_df = exp_data.get("val_comparison_df", pd.DataFrame())
    test_df = exp_data.get("test_comparison_df", pd.DataFrame())
    eval_info = exp_data.get("evaluation", {})

    if val_df.empty:
        st.info("No model comparison data available for this run.")
    else:
        st.subheader("Validation Set Model Comparison Table")
        st.dataframe(val_df, use_container_width=True)

        if not test_df.empty:
            st.subheader("Held-Out Test Set Performance (Best Model vs Baselines)")
            st.dataframe(test_df, use_container_width=True)

        st.markdown("---")
        st.subheader("95% Bootstrap Confidence Intervals")
        boots = eval_info.get("bootstrap_intervals", {})
        if boots:
            b_records = []
            for m, ci in boots.items():
                b_records.append({
                    "Model": m,
                    "Qini Score (95% CI)": f"{ci.get('qini_score_ci', [0, 0])[0]:.4f} to {ci.get('qini_score_ci', [0, 0])[1]:.4f}",
                    "AUUC (95% CI)": f"{ci.get('auuc_ci', [0, 0])[0]:.4f} to {ci.get('auuc_ci', [0, 0])[1]:.4f}",
                })
            st.table(pd.DataFrame(b_records))


# --- PAGE 4: CUSTOMER TARGETING & SEGMENTATION ---
elif nav_selection == "4. Customer Targeting & Segmentation":
    st.title("Customer Segmentation: The Four Causal Archetypes")
    st.markdown("Prioritize Persuadables while suppressing Sure Things, Lost Causes, and Sleeping Dogs.")

    segs = exp_data.get("segments", [])
    if not segs:
        st.info("No customer segment data available for this run.")
    else:
        df_seg = pd.DataFrame(segs)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig_pie = px.pie(
                df_seg,
                names="segment_name",
                values="count",
                title="Customer Cohort Distribution",
                color="segment_name",
                color_discrete_map={
                    "Persuadables": "#2ca02c",
                    "Sure Things": "#1f77b4",
                    "Lost Causes": "#7f7f7f",
                    "Sleeping Dogs": "#d62728",
                },
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = px.bar(
                df_seg,
                x="segment_name",
                y="avg_estimated_uplift",
                color="segment_name",
                title="Average Estimated Uplift by Segment",
                labels={"avg_estimated_uplift": "Estimated Uplift tau(X)", "segment_name": "Segment"},
            )
            fig_bar.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Actionable Targeting Guide")
        for s in segs:
            with st.expander(f"{s['segment_name']} ({s['percentage']}% of population, {s['count']:,} customers)"):
                st.markdown(f"- **Recommendation**: `{s['targeting_recommendation']}`")
                st.markdown(f"- **Average Baseline Conversion Rate P(Y(0))**: {s['avg_predicted_baseline']:.4f}")
                st.markdown(f"- **Average Treated Conversion Rate P(Y(1))**: {s['avg_predicted_treated']:.4f}")
                st.markdown(f"- **Estimated CATE tau(x)**: {s['avg_estimated_uplift']:+.5f}")
                st.markdown(f"- **Policy Action**: {s['action_description']}")


# --- PAGE 5: BUSINESS SIMULATOR ---
elif nav_selection == "5. Business Simulator & Policy":
    st.title("Budget-Constrained Campaign Simulator")
    st.markdown("Simulate financial ROI under different budget, coupon cost, and margin parameters.")

    col1, col2, col3 = st.columns(3)
    sim_budget = col1.slider("Campaign Budget ($)", min_value=1000.0, max_value=50000.0, value=10000.0, step=1000.0)
    sim_cost = col2.slider("Intervention Cost per Customer ($)", min_value=0.50, max_value=10.0, value=2.50, step=0.25)
    sim_margin = col3.slider("Contribution Margin per Conversion ($)", min_value=10.0, max_value=150.0, value=50.0, step=5.0)

    # Dynamic interactive simulation
    opt = PolicyOptimizer(default_cost=sim_cost, default_margin=sim_margin)
    # Generate mock distribution mirroring test set uplift
    synthetic_uplifts = np.random.normal(0.045, 0.04, 10000)
    synthetic_props = np.random.beta(2, 20, 10000)

    sim_df = opt.compare_targeting_policies(
        uplift_scores=synthetic_uplifts,
        response_propensity_scores=synthetic_props,
        budget=sim_budget,
        cost_per_intervention=sim_cost,
        margin_per_conversion=sim_margin,
    )

    st.subheader(f"Simulation Results for ${sim_budget:,.0f} Budget")
    st.dataframe(sim_df, use_container_width=True)

    fig_sim = px.bar(
        sim_df,
        x="policy_name",
        y="expected_incremental_profit",
        color="policy_name",
        title="Projected Incremental Profit by Policy",
        labels={"expected_incremental_profit": "Net Profit ($)", "policy_name": "Policy"},
    )
    fig_sim.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_sim, use_container_width=True)

    # Sensitivity Heatmap / Table
    sens_df = exp_data.get("sensitivity_df", pd.DataFrame())
    if not sens_df.empty:
        st.subheader("Multidimensional Sensitivity Grid")
        st.dataframe(sens_df.head(20), use_container_width=True)


# --- PAGE 6: DECISION ASSISTANT ---
elif nav_selection == "6. Causal Decision Assistant":
    st.title("Grounded Causal Decision Assistant")
    st.markdown("Ask natural language business and causal inference questions. Backed by verified documentation and live experiment metrics.")

    assistant = DecisionAssistant(artifacts_dir="artifacts")

    # Quick prompt shortcuts
    st.markdown("**Suggested Questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    q1 = q_col1.button("Which targeting policy performed best?")
    q2 = q_col2.button("How does uplift compare to propensity?")
    q3 = q_col3.button("What causal assumptions are required?")

    default_prompt = ""
    if q1:
        default_prompt = "Which targeting policy performed best on held-out evaluation?"
    elif q2:
        default_prompt = "How does uplift targeting compare with response-propensity targeting?"
    elif q3:
        default_prompt = "What assumptions are required for causal interpretation?"

    user_query = st.text_input("Enter your business or causal question:", value=default_prompt)

    if user_query:
        with st.spinner("Analyzing experiment results and retrieving causal documentation..."):
            resp = assistant.answer_query(user_query, experiment_id=selected_exp)

            st.markdown("### Executive Answer")
            st.success(resp.direct_answer)

            if resp.validated_metrics:
                st.markdown("### Live Experiment Metrics")
                st.json(resp.validated_metrics)

            st.markdown("### Required Causal Assumptions")
            for a in resp.causal_assumptions:
                st.markdown(f"- {a}")

            st.markdown("### Recommended Next Experiment")
            st.info(resp.next_recommended_experiment)

            with st.expander("Cited Documentation Sources"):
                for src in resp.cited_sources:
                    st.markdown(f"- **{src['title']}** (ID: `{src['doc_id']}`) — Relevance: `{src['relevance_score']}`")

