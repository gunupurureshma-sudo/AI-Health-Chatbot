import streamlit as st
import pandas as pd
import numpy as np
import csv
import random
import re
import warnings
import tempfile
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
from fpdf import FPDF

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Health ChatBot", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00b4d8;
    }
    .result-card {
        background: linear-gradient(135deg, #1a2a1a, #1e3a1e);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        border-left: 4px solid #2ecc71;
    }
    .doctor-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #9b59b6;
    }
    .pharmacy-card {
        background: linear-gradient(135deg, #1a2a2a, #1e3535);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1abc9c;
    }
    .header-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #00b4d8, #0077b6, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .subtitle {
        text-align: center;
        color: #8892a4;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #00b4d8;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #00b4d8;
    }
    .metric-label {
        font-size: 14px;
        color: #8892a4;
        margin-top: 5px;
    }
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #00b4d8;
        margin: 20px 0 10px 0;
        border-bottom: 2px solid #00b4d8;
        padding-bottom: 8px;
    }
    .call-btn {
        display: inline-block;
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        color: white;
        padding: 8px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        font-size: 14px;
        margin: 5px 2px;
    }
    .map-btn {
        display: inline-block;
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        padding: 8px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        font-size: 14px;
        margin: 5px 2px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    training = pd.read_csv('Data/Training.csv')
    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    training = training.loc[:, ~training.columns.duplicated()]
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.33, random_state=42
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}
    return model, le, cols, symptoms_dict, training


@st.cache_resource
def load_data():
    description_list = {}
    precautionDictionary = {}
    with open('MasterData/symptom_Description.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                description_list[row[0]] = row[1]
    with open('MasterData/symptom_precaution.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    return description_list, precautionDictionary


@st.cache_resource
def load_doctors():
    return pd.read_csv('Data/doctors.csv')


@st.cache_resource
def load_pharmacies():
    return pd.read_csv('Data/pharmacies.csv')


model, le, cols, symptoms_dict, training = load_model()
description_list, precautionDictionary = load_data()
doctors_df = load_doctors()
pharmacies_df = load_pharmacies()


# ------------------ Symptom Extraction ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "loose motion": "diarrhea",
    "high temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}


def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(
            word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8
        )
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))


def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence


# ------------------ Google Maps URL ------------------
def get_maps_url(lat, lng, name):
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"


def get_directions_url(lat, lng):
    return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"


# ------------------ PDF Generator ------------------
def generate_pdf(name, age, gender, symptoms_list, disease, confidence, description, precautions):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 22)
    pdf.set_text_color(0, 119, 182)
    pdf.cell(0, 15, "AI Health ChatBot - Health Report", ln=True, align="C")
    pdf.set_draw_color(0, 180, 216)
    pdf.line(10, 25, 200, 25)
    pdf.ln(8)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 119, 182)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Name   : " + str(name), ln=True)
    pdf.cell(0, 8, "Age    : " + str(age), ln=True)
    pdf.cell(0, 8, "Gender : " + str(gender), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 119, 182)
    pdf.cell(0, 10, "Detected Symptoms", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, ", ".join(symptoms_list), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 119, 182)
    pdf.cell(0, 10, "Diagnosis Result", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Predicted Disease : " + str(disease), ln=True)
    pdf.cell(0, 8, "Confidence Score  : " + str(confidence) + "%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 119, 182)
    pdf.cell(0, 10, "About the Condition", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    clean_desc = description.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, clean_desc)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 119, 182)
    pdf.cell(0, 10, "Suggested Precautions", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    for i, prec in enumerate(precautions, 1):
        pdf.cell(0, 8, str(i) + ". " + str(prec), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Note: This is AI-generated. Please consult a real doctor.", ln=True, align="C")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name


# ------------------ Quotes ------------------
quotes = [
    "Health is wealth, take care of yourself.",
    "A healthy outside starts from the inside.",
    "Every day is a chance to get stronger.",
    "Take a deep breath, your health matters.",
    "Remember, self-care is not selfish."
]

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("<h2 style='color:#00b4d8;'>🏥 HealthBot</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8892a4;'>AI-Powered Healthcare Assistant</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.selectbox("Navigate To", [
        "🏠 Home & Diagnosis",
        "👨‍⚕️ Find Doctors",
        "💊 Find Pharmacies",
        "📅 Book Appointment"
    ])
    st.markdown("---")
    st.markdown("<p style='color:#8892a4; font-size:12px;'>Built with love using Python & ML</p>", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown("<div class='header-title'>🏥 AI Health ChatBot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your personal AI-powered health assistant</div>", unsafe_allow_html=True)
st.markdown("---")


# ==================== PAGE 1 - DIAGNOSIS ====================
if page == "🏠 Home & Diagnosis":
    st.markdown("<div class='section-title'>👤 Patient Information</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Full Name")
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("<div class='section-title'>🤒 Describe Your Symptoms</div>", unsafe_allow_html=True)
    symptoms_input = st.text_area(
        "Type your symptoms below",
        placeholder="e.g. I have fever and headache and stomach pain",
        height=100
    )
    col1, col2 = st.columns(2)
    with col1:
        num_days = st.slider("Days with symptoms", 1, 30, 3)
    with col2:
        severity = st.slider("Severity (1=mild, 10=severe)", 1, 10, 5)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Disease"):
        if not name:
            st.warning("Please enter your name.")
        elif not symptoms_input:
            st.warning("Please describe your symptoms.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                symptoms_list = extract_symptoms(symptoms_input, cols)

            if not symptoms_list:
                st.error("Could not detect symptoms. Please add more details.")
            else:
                st.success("Detected: " + ", ".join(symptoms_list))
                disease, confidence = predict_disease(symptoms_list)

                st.markdown("<div class='section-title'>🩺 Diagnosis Result</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        "<div class='metric-card'>"
                        "<div class='metric-value'>" + disease + "</div>"
                        "<div class='metric-label'>Predicted Disease</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        "<div class='metric-card'>"
                        "<div class='metric-value'>" + str(confidence) + "%</div>"
                        "<div class='metric-label'>Confidence Score</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                desc = description_list.get(disease, "No description available.")
                st.markdown(
                    "<div class='result-card'>"
                    "<h4>About " + disease + "</h4>"
                    "<p>" + desc + "</p>"
                    "</div>",
                    unsafe_allow_html=True
                )

                if disease in precautionDictionary:
                    precs = precautionDictionary[disease]
                    st.markdown(
                        "<div class='card'>"
                        "<h4>Suggested Precautions</h4>"
                        "<p>1. " + precs[0] + "</p>"
                        "<p>2. " + precs[1] + "</p>"
                        "<p>3. " + precs[2] + "</p>"
                        "<p>4. " + precs[3] + "</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )

                if severity >= 7 or num_days >= 7:
                    st.error("Symptoms seem serious! Please consult a doctor immediately!")
                else:
                    st.success("Follow the precautions and monitor your symptoms.")

                st.markdown("<br>", unsafe_allow_html=True)
                precs_for_pdf = precautionDictionary.get(disease, ["", "", "", ""])
                pdf_path = generate_pdf(
                    name, age, gender, symptoms_list,
                    disease, confidence, desc, precs_for_pdf
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📄 Download Health Report (PDF)",
                        data=f,
                        file_name=name + "_health_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                st.markdown("---")
                st.info("Go to Find Doctors or Book Appointment from the sidebar!")
                st.markdown(
                    "<p style='color:#00b4d8; font-size:18px;'>💡 " + random.choice(quotes) + "</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='color:#2ecc71;'>Thank you, " + name + "! Wishing you good health!</h3>",
                    unsafe_allow_html=True
                )


# ==================== PAGE 2 - DOCTORS ====================
elif page == "👨‍⚕️ Find Doctors":
    st.markdown("<div class='section-title'>👨‍⚕️ Find Nearby Doctors</div>", unsafe_allow_html=True)
    specialization_filter = st.selectbox(
        "Filter by Specialization",
        ["All"] + list(doctors_df['specialization'].unique())
    )
    if specialization_filter == "All":
        filtered = doctors_df
    else:
        filtered = doctors_df[doctors_df['specialization'] == specialization_filter]

    for _, doc in filtered.iterrows():
        maps_url = get_maps_url(doc['lat'], doc['lng'], doc['name'])
        directions_url = get_directions_url(doc['lat'], doc['lng'])
        call_url = "tel:" + str(doc['contact'])

        st.markdown(
            "<div class='doctor-card'>"
            "<h3>👨‍⚕️ " + doc['name'] + "</h3>"
            "<p>🩺 <b>Specialization:</b> " + doc['specialization'] + "</p>"
            "<p>🏥 <b>Clinic:</b> " + doc['clinic'] + "</p>"
            "<p>📞 <b>Contact:</b> " + str(doc['contact']) + "</p>"
            "<p>📍 <b>Location:</b> " + doc['location'] + "</p>"
            "<p>🕐 <b>Available Slots:</b> " + doc['slots'] + "</p>"
            "<br>"
            "<a href='" + call_url + "' class='call-btn'>📞 Call Doctor</a>"
            "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 View on Maps</a>"
            "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions</a>"
            "</div>",
            unsafe_allow_html=True
        )

    # Show all doctors on embedded map
    st.markdown("<div class='section-title'>🗺️ Doctors Location Map</div>", unsafe_allow_html=True)
    map_html = """
    <iframe
        width="100%"
        height="400"
        frameborder="0"
        style="border-radius:16px; border:2px solid #00b4d8;"
        src="https://www.google.com/maps/embed/v1/search?key=AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY&q=doctors+in+Hyderabad"
        allowfullscreen>
    </iframe>
    """
    st.markdown(map_html, unsafe_allow_html=True)


# ==================== PAGE 3 - PHARMACIES ====================
elif page == "💊 Find Pharmacies":
    st.markdown("<div class='section-title'>💊 Nearby Pharmacies</div>", unsafe_allow_html=True)
    for _, pharmacy in pharmacies_df.iterrows():
        maps_url = get_maps_url(pharmacy['lat'], pharmacy['lng'], pharmacy['name'])
        directions_url = get_directions_url(pharmacy['lat'], pharmacy['lng'])
        call_url = "tel:" + str(pharmacy['contact'])

        st.markdown(
            "<div class='pharmacy-card'>"
            "<h3>💊 " + pharmacy['name'] + "</h3>"
            "<p>📍 <b>Address:</b> " + pharmacy['address'] + "</p>"
            "<p>📞 <b>Contact:</b> " + str(pharmacy['contact']) + "</p>"
            "<br>"
            "<a href='" + call_url + "' class='call-btn'>📞 Call Pharmacy</a>"
            "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 View on Maps</a>"
            "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions</a>"
            "</div>",
            unsafe_allow_html=True
        )

    # Show all pharmacies on embedded map
    st.markdown("<div class='section-title'>🗺️ Pharmacies Location Map</div>", unsafe_allow_html=True)
    map_html = """
    <iframe
        width="100%"
        height="400"
        frameborder="0"
        style="border-radius:16px; border:2px solid #1abc9c;"
        src="https://www.google.com/maps/embed/v1/search?key=AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY&q=pharmacy+in+Hyderabad"
        allowfullscreen>
    </iframe>
    """
    st.markdown(map_html, unsafe_allow_html=True)


# ==================== PAGE 4 - BOOKING ====================
elif page == "📅 Book Appointment":
    st.markdown("<div class='section-title'>📅 Book Doctor Appointment</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Your Name")
    with col2:
        patient_contact = st.text_input("Your Contact Number")

    selected_doctor = st.selectbox("Select Doctor", doctors_df['name'].tolist())
    doctor_info = doctors_df[doctors_df['name'] == selected_doctor].iloc[0]

    maps_url = get_maps_url(doctor_info['lat'], doctor_info['lng'], doctor_info['name'])
    directions_url = get_directions_url(doctor_info['lat'], doctor_info['lng'])
    call_url = "tel:" + str(doctor_info['contact'])

    st.markdown(
        "<div class='doctor-card'>"
        "<p>🏥 <b>Clinic:</b> " + doctor_info['clinic'] + "</p>"
        "<p>🩺 <b>Specialization:</b> " + doctor_info['specialization'] + "</p>"
        "<p>📍 <b>Location:</b> " + doctor_info['location'] + "</p>"
        "<p>📞 <b>Contact:</b> " + str(doctor_info['contact']) + "</p>"
        "<br>"
        "<a href='" + call_url + "' class='call-btn'>📞 Call Doctor</a>"
        "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 View on Maps</a>"
        "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions</a>"
        "</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        selected_slot = st.selectbox("Select Time Slot", doctor_info['slots'].split(','))
    with col2:
        appointment_date = st.date_input("Select Date")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Confirm Appointment"):
        if not patient_name:
            st.warning("Please enter your name.")
        elif not patient_contact:
            st.warning("Please enter your contact number.")
        else:
            st.balloons()
            st.markdown(
                "<div class='result-card'>"
                "<h2>🎉 Appointment Confirmed!</h2>"
                "<p><b>Patient:</b> " + patient_name + "</p>"
                "<p><b>Contact:</b> " + patient_contact + "</p>"
                "<p><b>Doctor:</b> " + selected_doctor + "</p>"
                "<p><b>Specialization:</b> " + doctor_info['specialization'] + "</p>"
                "<p><b>Clinic:</b> " + doctor_info['clinic'] + "</p>"
                "<p><b>Date:</b> " + str(appointment_date) + "</p>"
                "<p><b>Time:</b> " + selected_slot + "</p>"
                "<p><b>Doctor Contact:</b> " + str(doctor_info['contact']) + "</p>"
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions to Clinic</a>",
                unsafe_allow_html=True
            )