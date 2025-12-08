import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config, data_prep
from src.loyalty_scoring import load_models, compute_loyalty_for_row, score_all_customers
from src.scenario_simulator import simulate_scenario
from src.explainability import (
    global_feature_importance_subscription,
    global_feature_importance_frequency,
)


@st.cache_data
def load_data():
    return data_prep.load_clean()


@st.cache_resource
def load_loyalty_models():
    return load_models()


def main():
    st.set_page_config(page_title="Subscription Loyalty Risk Radar", layout="wide")
    st.title("Subscription-Loyalty-Risk-Radar")


    df = load_data()
    models = load_loyalty_models()

    tab_overview, tab_customer, tab_segments, tab_scenarios, tab_importance = st.tabs(
        [
            "Overview",
            "Customer Explorer",
            "Segment Insights",
            "Scenario Simulator",
            "Feature Importance",
        ]
    )

    with tab_overview:
        st.subheader("Dataset snapshot")
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        st.dataframe(df.head())

        st.markdown("---")
        st.subheader("Loyalty distribution (after scoring)")
        if st.button("Compute loyalty scores for all customers"):
            scored = score_all_customers(df)
            st.session_state["scored_df"] = scored
            st.success("Scored customers computed.")
            st.dataframe(scored.head())

            if "loyalty_risk" in scored.columns:
                st.write("Loyalty risk summary:")
                st.write(scored["loyalty_risk"].describe())

    with tab_customer:
        st.subheader("Customer Explorer")

        customer_ids = df[config.ID_COLUMN].unique().tolist()
        selected_id = st.selectbox("Select Customer ID", customer_ids)

        row = df[df[config.ID_COLUMN] == selected_id].iloc[0]
        st.write("Raw attributes:")
        st.json(row.to_dict())

        metrics = compute_loyalty_for_row(row, models)
        st.markdown("### Loyalty metrics")
        st.metric("Subscription probability", f"{metrics['p_subscribe']:.2f}")
        st.metric("Predicted frequency score", f"{metrics['predicted_frequency_score']:.2f}")
        st.metric("Loyalty index", f"{metrics['loyalty_index']:.2f}")
        st.metric("Loyalty risk (0–100)", f"{metrics['loyalty_risk']:.1f}")

    with tab_segments:
        st.subheader("Segment Insights")
        segment_col = st.selectbox(
            "Segment by column",
            ["Gender", "Season", "Category", "Payment Method", "Shipping Type"],
        )

        if "scored_df" not in st.session_state:
            st.info("Compute scores in the Overview tab first.")
        else:
            scored_df = st.session_state["scored_df"]
            grouped = scored_df.groupby(segment_col)["loyalty_risk"].mean().sort_values()
            st.bar_chart(grouped)

            st.write("Average loyalty risk by segment:")
            st.dataframe(grouped.reset_index().rename(columns={"loyalty_risk": "avg_loyalty_risk"}))

    with tab_scenarios:
        st.subheader("Scenario Simulator")

        customer_ids = df[config.ID_COLUMN].unique().tolist()
        selected_id = st.selectbox("Customer ID for scenario", customer_ids, key="scenario_customer")

        base_row = df[df[config.ID_COLUMN] == selected_id].iloc[0]

        st.write("Base attributes:")
        st.json(base_row.to_dict())

        st.markdown("### Hypothetical changes")
        new_shipping = st.selectbox(
            "Shipping Type", df["Shipping Type"].unique().tolist(), index=0
        )
        new_discount = st.selectbox(
            "Discount Applied", df["Discount Applied"].unique().tolist(), index=0
        )
        new_promo = st.selectbox(
            "Promo Code Used", df["Promo Code Used"].unique().tolist(), index=0
        )

        if st.button("Run scenario"):
            changes = {
                "Shipping Type": new_shipping,
                "Discount Applied": new_discount,
                "Promo Code Used": new_promo,
            }
            result = simulate_scenario(base_row, changes, models)
            st.write("Before:")
            st.json(result["before"])
            st.write("After:")
            st.json(result["after"])
            st.write("Delta:")
            st.json(result["delta"])

    with tab_importance:
        st.subheader("Global Feature Importance")


        if st.button("Compute subscription feature importance"):
            imp_sub = global_feature_importance_subscription(top_n=20)
            st.dataframe(imp_sub)

        if st.button("Compute frequency feature importance"):
            imp_freq = global_feature_importance_frequency(top_n=20)
            st.dataframe(imp_freq)


if __name__ == "__main__":
    main()
