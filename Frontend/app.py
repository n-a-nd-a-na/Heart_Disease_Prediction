import streamlit as st
import requests
import plotly.graph_objects as go

st.title('Heart Disease Risk Prediction App')

st.write("Enter the patient's details to predict the risk of Heart Disease")


with st.form('patient_form'):
    st.subheader('Patient Information')

    age = st.number_input('Age',min_value = 0,max_value = 110,value = None,placeholder='Please enter a valid age between 1 and 110')

    sex = st.selectbox('Gender',['M','F'])
    st.info("""
        - **M** - Male
        - **F** - Female
            """)

    chest_pain = st.selectbox('Chest Pain Type',['ASY','ATA','NAP','TA'])
    st.info("""
        - **TA** - Typical Angina
        - **ATA** - Atypical Angina
        - **NAP** - Non-Anginal Pain
        - **ASY** - Asymptomatic
            """)

    resting_bp = st.number_input('Resting Blood Pressure',min_value = 50,max_value = 250,value = None,placeholder='Please enter a valid value between 50 and 250')

    cholesterol = st.number_input('Cholesterol Level',min_value = 0,max_value = 700,value = None,placeholder='Please enter a valid value between 0 and 700')

    fasting_bs = st.selectbox('Fasting Blood Sugar',[0,1])
    st.info("""
           - **0** - No
           - **1** - Yes
               """)

    resting_ecg = st.selectbox('Resting ECG Result',['Normal','ST','LVH'])
    st.info("""
        - **Normal** - Normal ECG
        - **ST** - ST-T wave abnormality
        - **LVH** - Left Ventricular Hypertrophy
        """)

    max_hr = st.number_input('Maximum Heart Rate ',min_value = 50,max_value = 250,value = None,placeholder='Please enter a valid value between 50 and 250')

    exercise_angina = st.selectbox('Chest Pain During Exercise',['N','Y'])
    st.info("""
               - **N** - No
               - **Y** - Yes
                   """)

    oldpeak = st.number_input('ST Segment Depression',min_value = -5.0,max_value = 10.0,value = None,step = 0.1,placeholder='Please enter a valid age between -5.0 and 10.0')

    st_slope = st.selectbox('ST Segment Direction During Exercise',['Up','Flat','Down'])
    st.info("""
        - **Up** - Upsloping
        - **Flat** - Flat
        - **Down** - Downsloping
        """)

    submitted = st.form_submit_button('Predict Risk')

if submitted:
    patient_data = { "Age": age,"Sex": sex,"ChestPainType": chest_pain,"RestingBP": resting_bp,"Cholesterol": cholesterol,"FastingBS": fasting_bs,
                     "RestingECG": resting_ecg,"MaxHR": max_hr,"ExerciseAngina": exercise_angina,"Oldpeak": oldpeak,"ST_Slope": st_slope}


    try:
        response = requests.post("http://Backend:8000/predict",json=patient_data)


        if response.status_code == 200:
            result = response.json()

            prediction = result['prediction']
            risk = result['risk_percentage']

            st.subheader('Predicton Result')

            if prediction == 0:
                st.success('No Heart Disease Detected')

            else:
                st.error('Heart Disease Detected')

            fig = go.Figure(go.Indicator(mode = 'gauge+number',value = risk,number={'suffix': '%','font':{'size': 50},'valueformat': '.1f'},
                                         title = {'text': 'Heart Disease Risk'},gauge = {'axis' : {'range' : [0,100],'tickvals':[0,20,40,60,80,100]},
                                        'bar':{'color':'rgba(0,0,0,0)'},'steps' :[{'range':[0,40],'color' : 'green'},{'range':[40,70],'color' : 'yellow'},
                                        {'range':[70,100],'color' : 'red'}],'threshold':{'line':{'color':'black','width':5},
                                        'thickness':0.75,'value': risk}}))

            st.plotly_chart(fig)

        else:
            st.error('Prediction Failed')

    except requests.exceptions.ConnectionError:

        st.error('Cannot connect to FastAPI , Please start the FastAPI server')

