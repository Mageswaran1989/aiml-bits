import pandas as pd
import streamlit as st
from joblib import load
st.title('Android Malware Detection System')

clf = load("data/model_store/rf_corr_engd_clf_pipeline.joblib")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    res = clf.predict(df)

    if df["class"].values[0] == 0:
        st.warning("Sample file contains no malware")
    else:
        st.warning(":red[Sample file contains a malware]")

    if res[0] == 1:
        st.success('MDS Alert: :red[Its a Malware]', icon="⚠️")
    else:
        st.success('MDS Alert: :green[Its not a Malware]', icon="✅")



