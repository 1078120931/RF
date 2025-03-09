import streamlit as st
import shap
import joblib
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    best_model = joblib.load('./rf.pkl')

    class Subject:
        def __init__(self, ICU_stay, DBC, pan_MDRO, blood_MDRO, OF_time):
            self.ICU_stay = ICU_stay
            self.DBC = DBC
            self.pan_MDRO = pan_MDRO
            self.blood_MDRO = blood_MDRO
            self.OF_time = OF_time
           
        def make_predict(self):
            subject_data = {
                "ICU_stay": [self.ICU_stay],
                "DBC": [self.DBC],
                "pan_MDRO": [self.pan_MDRO],
                "blood_MDRO": [self.blood_MDRO],
                "OF_time": [self.OF_time]
                }

            # Create a DataFrame
            df_subject = pd.DataFrame(subject_data)

            # Make the prediction
            prediction = best_model.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>The model predicts the risk of hemorrhage is {adjusted_prediction} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(best_model)
            shap_values = explainer.shap_values(df_subject)
            # 力图
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            # 瀑布图
            # ex = shap.Explanation(shap_values[1][0, :], explainer.expected_value[1], df_subject.iloc[0, :])
            # shap.waterfall_plot(ex)
            st.pyplot(plt.gcf())

    st.set_page_config(page_title='IPN hemorrhage')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Predicting IPN hemorrhage</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    ICU_stay=st.selectbox("ICU stay (No = 0, Yes = 1)", [0, 1], index=0)
    DBC = st.selectbox("DBC (IPN only = 0, MCAP=1, SCAP = 2)", [0, 1, 2], index=0)
    pan_MDRO=st.selectbox("Pus MDRO infection (No = 0, Yes = 1)", [0, 1], index=0)
    blood_MDRO = st.selectbox("Blood MDRO infection (No = 0, Yes = 1)", [0, 1], index=0)
    OF_time = st.number_input("OF (days)", value=0)
    

    if st.button(label="Submit"):
        user = Subject(ICU_stay, DBC, pan_MDRO, blood_MDRO, OF_time)
        user.make_predict()

main()
