import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="BharathonAI/tourism_newplan_adoption_model", filename="best_toursim_opt_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism new plan opt prediction App")
st.write("""
This application predicts the likelihood of a user opting to the new Tourism plan.
Please enter the sensor and configuration data below to get a prediction.
""")

age = st.number_input("Age",min_value=18,max_value=100,value=30,step=1,help="Enter your age")
monthly_income = st.number_input("Monthly Income ($)",min_value=0,max_value=1000000,value=5000,step=100,help="Enter your monthly income in dollars")
number_of_trips = st.number_input("Number of Trips",min_value=0,max_value=100,value=1,step=1,help="Number of trips taken in the past year")
type_of_contact = st.selectbox("Type of Contact",options=["Self Enquiry", "Company Invited"],help="How did you initiate contact?")
city_tier = st.selectbox("City Tier",options=[1, 2, 3],help="Select your city tier")
occupation = st.selectbox("Occupation",options=["Salaried", "Small Business", "Large Business", "Free Lancer"],help="Select your occupation type"
    )
gender = st.selectbox("Gender",options=["Male", "Female"],help="Select your gender")
number_of_person_visiting = st.selectbox(
        "Number of Person Visiting",
        options=[1, 2, 3, 4, 5],
        help="How many people will be traveling?"
    )
    
preferred_property_star = st.selectbox(
    "Preferred Property Star",
    options=[3, 4, 5],
    help="Select preferred hotel star rating"
)

marital_status = st.selectbox(
    "Marital Status",
    options=["Single", "Married", "Divorced", "Unmarried"],
    help="Select your marital status"
)
passport = st.selectbox(
    "Passport",
    options=["Yes", "No"],
    help="Do you have a passport?"
)

own_car = st.selectbox(
    "Own Car",
    options=["Yes", "No"],
    help="Do you own a car?"
)

number_of_children_visiting = st.selectbox(
    "Number of Children Visiting",
    options=[0, 1, 2, 3, 4],
    help="How many children will be traveling?"
)
designation = st.selectbox(
        "Designation",
        options=["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        help="Select your job designation"
    )

passport_encoded = 1 if passport == "Yes" else 0
own_car_encoded = 1 if own_car == "Yes" else 0

# Assemble input into DataFrame
input_data = pd.DataFrame([{
            'Age': age,
            'MonthlyIncome': monthly_income,
            'NumberOfTrips': number_of_trips,
            'TypeofContact': type_of_contact,
            'CityTier': city_tier,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': number_of_person_visiting,
            'PreferredPropertyStar': preferred_property_star,
            'MaritalStatus': marital_status,
            'Passport': passport_encoded,
            'OwnCar': own_car_encoded,
            'NumberOfChildrenVisiting': number_of_children_visiting,
            'Designation': designation
        }
])


if st.button("Predict Plan Opt"):
    prediction = model.predict(input_data)[0]
    result = "Plan will be Opted" if prediction == 1 else "Plan will not be Opted"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
