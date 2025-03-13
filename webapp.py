import streamlit as st
import shap
import joblib
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    best_model = joblib.load('./xgb.pkl')

    class Subject:
        def __init__(self, ICU_stay, pan_MDRO, blood_inf, OF_num, pancreatic_fis, surgery_app):
            self.ICU_stay = ICU_stay
            self.pan_MDRO = pan_MDRO
            self.blood_inf = blood_inf
            self.OF_num = OF_num
            self.pancreatic_fis = pancreatic_fis
            self.surgery_app = surgery_app
           
        def make_predict(self):
            subject_data = {
                "ICU_stay": [self.ICU_stay],
                "pan_MDRO": [self.pan_MDRO],
                "blood_inf": [self.blood_inf],
                "OF_num": [self.OF_num],
                "pancreatic_fis": [self.pancreatic_fis],
                "surgery_app": [self.surgery_app]
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
            shap.force_plot(explainer.expected_value, shap_values[0, :], df_subject.iloc[0, :], matplotlib=True)
            # 瀑布图
            #ex = shap.Explanation(shap_values[0, :], explainer.expected_value, df_subject.iloc[0, :])
            #shap.waterfall_plot(ex)
            st.pyplot(plt.gcf())

    st.set_page_config(page_title='IPN hemorrhage')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Predicting IPN hemorrhage</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    ICU_stay=st.number_input("ICU stays (days)", value=0)
    pan_MDRO=st.selectbox("Pus MDRO infection (No = 0, Yes = 1)", [0, 1], index=0)
    blood_inf = st.selectbox("Blood infection (No = 0, Yes = 1)", [0, 1], index=0)
    OF_num = st.selectbox("OF (No = 0, Single = 1, Multi=2)", [0, 1, 2], index=0)
    pancreatic_fis = st.selectbox("pancreatic fistula (No = 0, Yes = 1)", [0, 1], index=0)
    surgery_app = st.selectbox("Surgery opproach (UP = 1, Down = 2)", [1, 2], index=0)
    

    if st.button(label="Submit"):
        user = Subject(ICU_stay, pan_MDRO, blood_inf, OF_num, pancreatic_fis, surgery_app)
        user.make_predict()

main()
