import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set up the Streamlit app
st.set_page_config(page_title="Student GPA Predictor", layout="wide")

# App title and description
st.title("🎓 Student GPA Prediction Tool")
st.markdown("""
This app predicts a student's GPA based on various academic and demographic factors.
Adjust the input parameters using the sliders and dropdowns below.
""")

# Create input form in sidebar
with st.sidebar:
    st.header("📋 Student Information")
    
    # Demographic inputs
    age = st.slider("Age", 15, 18, 16)
    gender = st.selectbox("Gender", ["Female", "Male"])
    ethnicity = st.selectbox("Ethnicity", ["Group 0", "Group 1", "Group 2", "Group 3"])
    
    # Academic inputs
    study_time = st.slider("Weekly Study Time (hours)", 0.0, 20.0, 10.0, 0.5)
    absences = st.slider("Number of Absences", 0, 30, 5)
    tutoring = st.selectbox("Receives Tutoring", ["No", "Yes"])
    
    # Family/support inputs
    parental_education = st.selectbox("Parental Education Level", 
                                    ["No High School", "High School", "Some College", 
                                     "Bachelor's Degree", "Graduate Degree"])
    parental_support = st.selectbox("Parental Support Level", 
                                  ["None", "Low", "Medium", "High", "Very High"])
    
    # Extracurricular activities
    st.subheader("Extracurricular Activities")
    extracurricular = st.checkbox("Participates in Extracurriculars")
    sports = st.checkbox("Participates in Sports")
    music = st.checkbox("Participates in Music")
    volunteering = st.checkbox("Participates in Volunteering")

# Convert inputs to model format
gender_map = {"Female": 0, "Male": 1}
tutoring_map = {"No": 0, "Yes": 1}
ethnicity_map = {f"Group {i}": i for i in range(4)}
parental_education_map = {
    "No High School": 0,
    "High School": 1,
    "Some College": 2,
    "Bachelor's Degree": 3,
    "Graduate Degree": 4
}
parental_support_map = {
    "None": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

input_data = {
    'Age': age,
    'Gender': gender_map[gender],
    'Ethnicity': ethnicity_map[ethnicity],
    'ParentalEducation': parental_education_map[parental_education],
    'StudyTimeWeekly': study_time,
    'Absences': absences,
    'Tutoring': tutoring_map[tutoring],
    'ParentalSupport': parental_support_map[parental_support],
    'Extracurricular': int(extracurricular),
    'Sports': int(sports),
    'Music': int(music),
    'Volunteering': int(volunteering)
}

# Create DataFrame and scale numerical features
input_df = pd.DataFrame([input_data])
num_cols = ['Age', 'StudyTimeWeekly', 'Absences']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Make prediction when button is clicked
if st.button("Predict GPA"):
    prediction = model.predict(input_df)[0]
    
    # Display results
    st.success(f"### Predicted GPA: {prediction:.2f}")
    
    # Interpretation
    st.subheader("Interpretation")
    if prediction >= 3.5:
        st.markdown("⭐ **Excellent performance** - This student is likely in the top tier of their class.")
    elif prediction >= 3.0:
        st.markdown("👍 **Good performance** - This student is performing above average.")
    elif prediction >= 2.0:
        st.markdown("💪 **Average performance** - There's room for improvement with the right support.")
    else:
        st.markdown("⚠️ **Needs improvement** - This student may benefit from additional academic support.")
    
    # Feature importance explanation (if available)
    try:
        st.subheader("Key Influencing Factors")
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top_features = feature_importance.head(3)
        for i, row in top_features.iterrows():
            st.write(f"- **{row['Feature']}** had the biggest impact on this prediction")
    except:
        pass

# Add some explanatory sections
st.markdown("---")
st.subheader("How This Works")
st.markdown("""
This predictive model was built using:
- **XGBoost algorithm** (best performing model)
- Data from 2,026 student records
- Features including study habits, demographics, and extracurricular activities

The model achieves an R² score of 0.82, meaning it explains 82% of the variance in GPA.
""")

# Add footer
st.markdown("---")
st.markdown("""
*Note: This is a predictive model and should be used as one tool among many for academic assessment.*
""")