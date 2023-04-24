import pickle
from typing import List

import pandas as pd
import streamlit as st

"""
# Grid Stability Prediction App
"""


# Sidebar
def add_input_set(
    feature: str,
    min_value: float,
    max_value: float,
    value: float,
    node_names: List[str],
):
    assert len(node_names) > 0, "At least one string required in `node_names` list."
    assert all(
        [name != "" for name in node_names]
    ), "Names in `node_names` list can't be empty strings."

    inputs = [
        st.slider(node_name, min_value, max_value, value) for node_name in node_names
    ]
    data = {feature + str(i + 1): inputs[i] for i in range(len(node_names))}
    final_df = pd.DataFrame(data, index=[0])
    return final_df


with st.sidebar:
    st.header("Grid conditions")

    with st.expander("Response delay", expanded=False):
        st.write(
            "How long it takes for each node to adapt their production or consumption in seconds:"
        )
        p_delay_df = add_input_set("p_delay", 0.5, 10.0, 0.5, ["Producer"])
        c_delay_df = add_input_set(
            "c_delay", 0.5, 10.0, 5.0, ["Consumer1", "Consumer2", "Consumer3"]
        )

    with st.expander("Willingness to adapt", expanded=True):
        st.write(
            "Willingness of each node to adapt their consumption or production per second:"
        )
        p_adapt_df = add_input_set("p_adapt", 0.05, 1.0, 0.05, ["Producer"])
        c_adapt_df = add_input_set(
            "c_adapt", 0.05, 1.0, 0.5, ["Consumer1", "Consumer2", "Consumer3"]
        )

input_df = p_delay_df.join(c_delay_df).join(p_adapt_df).join(c_adapt_df)


# Result
st.subheader("Grid condition summary:")
st.write(input_df)

st.subheader("Predictions:")
clf = pickle.load(open("grid_clf.pkl", "rb"))
reg = pickle.load(open("grid_reg.pkl", "rb"))

clf_pred = clf.predict(input_df)[0]
reg_pred = reg.predict(input_df)[0]

stability_dict = {0: "unstable", 1: "stable"}
st.markdown(f"Best classifier's prediction: **{stability_dict.get(clf_pred)}**")
st.markdown(
    f"Best regressor's prediction: {round(reg_pred, 3)} (**{stability_dict.get(int(reg_pred < 0))}**)"
)