import streamlit as st
import pandas as pd
import numpy as np
import csv
import random
import re
import warnings
import tempfile
import sqlite3
import hashlib
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
    .card { background: linear-gradient(135deg, #1e2130, #252a3d); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 4px solid #00b4d8; }
    .result-card { background: linear-gradient(135deg, #1a2a1a, #1e3a1e); border-radius: 16px; padding: 24px; margin: 10px 0; border-left: 4px solid #2ecc71; }
    .doctor-card { background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 4px solid #9b59b6; }
    .pharmacy-card { background: linear-gradient(135deg, #1a2a2a, #1e3535); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 4px solid #1abc9c; }
    .hospital-card { background: linear-gradient(135deg, #2a1a1a, #3a1e1e); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 4px solid #e74c3c; }
    .top1-card { background: linear-gradient(135deg, #1a2a1a, #1e3a1e); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 6px solid #2ecc71; }
    .top2-card { background: linear-gradient(135deg, #2a2a1a, #3a3a1e); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 6px solid #f39c12; }
    .top3-card { background: linear-gradient(135deg, #2a1a2a, #3a1e3a); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 6px solid #9b59b6; }
    .login-card { background: linear-gradient(135deg, #1e2130, #252a3d); border-radius: 20px; padding: 40px; margin: 20px auto; border: 2px solid #00b4d8; max-width: 500px; }
    .header-title { font-size: 48px; font-weight: 800; background: linear-gradient(90deg, #00b4d8, #0077b6, #9b59b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 20px 0; }
    .subtitle { text-align: center; color: #8892a4; font-size: 18px; margin-bottom: 30px; }
    .metric-card { background: linear-gradient(135deg, #1e2130, #252a3d); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #00b4d8; }
    .metric-value { font-size: 32px; font-weight: 800; color: #00b4d8; }
    .metric-label { font-size: 14px; color: #8892a4; margin-top: 5px; }
    .section-title { font-size: 24px; font-weight: 700; color: #00b4d8; margin: 20px 0 10px 0; border-bottom: 2px solid #00b4d8; padding-bottom: 8px; }
    .call-btn { display: inline-block; background: linear-gradient(90deg, #2ecc71, #27ae60); color: white; padding: 8px 20px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 14px; margin: 5px 2px; }
    .map-btn { display: inline-block; background: linear-gradient(90deg, #e74c3c, #c0392b); color: white; padding: 8px 20px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 14px; margin: 5px 2px; }
    .chatbot-bubble { background: linear-gradient(135deg, #1e2130, #252a3d); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 4px solid #00b4d8; font-size: 18px; }
    .role-card { background: linear-gradient(135deg, #1e2130, #252a3d); border-radius: 20px; padding: 30px; margin: 10px; border: 2px solid #00b4d8; text-align: center; cursor: pointer; }
    .welcome-card { background: linear-gradient(135deg, #1a2a1a, #1e3a1e); border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 4px solid #2ecc71; }
</style>
""", unsafe_allow_html=True)


# ------------------ Database ------------------
def init_db():
    conn = sqlite3.connect('health_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, password TEXT,
        age INTEGER, gender TEXT, contact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS doctors_login (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, password TEXT,
        specialization TEXT, clinic TEXT, contact TEXT,
        slots TEXT, location TEXT, lat REAL, lng REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT, patient_contact TEXT, patient_email TEXT,
        doctor_name TEXT, clinic TEXT,
        date TEXT, slot TEXT, disease TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS hospitals_reg (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, address TEXT, contact TEXT,
        speciality TEXT, beds INTEGER, lat REAL, lng REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS pharmacies_reg (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, address TEXT, contact TEXT, lat REAL, lng REAL)''')
    conn.commit()
    conn.close()

init_db()


# ------------------ Password Hashing ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ------------------ Auth Functions ------------------
def signup_patient(name, email, password, age, gender, contact):
    try:
        conn = sqlite3.connect('health_app.db')
        c = conn.cursor()
        c.execute("INSERT INTO patients (name, email, password, age, gender, contact) VALUES (?, ?, ?, ?, ?, ?)",
            (name, email, hash_password(password), age, gender, contact))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Email already registered!"


def login_patient(email, password):
    conn = sqlite3.connect('health_app.db')
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE email=? AND password=?", (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user


def signup_doctor(name, email, password, specialization, clinic, contact, slots, location, lat, lng):
    try:
        conn = sqlite3.connect('health_app.db')
        c = conn.cursor()
        c.execute('''INSERT INTO doctors_login
            (name, email, password, specialization, clinic, contact, slots, location, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (name, email, hash_password(password), specialization, clinic, contact, slots, location, lat, lng))
        conn.commit()
        conn.close()
        return True, "Doctor account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Email already registered!"


def login_doctor(email, password):
    conn = sqlite3.connect('health_app.db')
    c = conn.cursor()
    c.execute("SELECT * FROM doctors_login WHERE email=? AND password=?", (email, hash_password(password)))
    doctor = c.fetchone()
    conn.close()
    return doctor


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
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=500, random_state=42)
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


# ------------------ Symptom Synonyms ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain", "belly pain": "stomach_pain", "tummy pain": "stomach_pain",
    "abdominal pain": "abdominal_pain", "chest pain": "chest_pain", "back pain": "back_pain",
    "joint pain": "joint_pain", "knee pain": "knee_pain", "body pain": "body_pain",
    "muscle pain": "muscle_pain", "body ache": "muscle_pain", "neck pain": "stiff_neck",
    "stiff neck": "stiff_neck", "high temperature": "high_fever", "temperature": "mild_fever",
    "fever": "high_fever", "feaver": "high_fever", "mild fever": "mild_fever",
    "loose motion": "diarrhea", "motions": "diarrhea", "loose stool": "diarrhea",
    "vomit": "vomiting", "throwing up": "vomiting", "feel like vomiting": "nausea",
    "acidity": "acidity", "heartburn": "acidity", "itching": "itching", "itchy": "itching",
    "skin rash": "skin_rash", "rash": "skin_rash", "yellow skin": "yellowing_of_skin",
    "jaundice": "yellowing_of_skin", "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness", "difficulty breathing": "breathlessness",
    "coughing": "cough", "dry cough": "cough", "cold": "runny_nose",
    "throat pain": "sore_throat", "sore throat": "sore_throat",
    "tired": "fatigue", "tiredness": "fatigue", "weakness": "weakness",
    "headache": "headache", "head pain": "headache", "migraine": "headache",
    "dizziness": "dizziness", "dizzy": "dizziness", "vertigo": "dizziness",
    "swollen joints": "swollen_joints", "swelling": "swollen_joints",
    "weight loss": "weight_loss", "loss of appetite": "loss_of_appetite",
    "frequent urination": "polyuria", "sweating": "sweating", "chills": "chills",
    "shivering": "chills", "nausea": "nausea", "fatigue": "fatigue",
    "constipation": "constipation", "painful walking": "painful_walking",
    "blurred vision": "blurred_and_distorted_vision",
}


def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text and mapped in list(all_symptoms):
            extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.82)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))


def predict_top3(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    top3_idx = np.argsort(pred_proba)[-3:][::-1]
    results = []
    for idx in top3_idx:
        disease = le.inverse_transform([idx])[0]
        confidence = round(pred_proba[idx] * 100, 2)
        results.append({"disease": disease, "confidence": confidence})
    return results


def get_maps_url(lat, lng):
    return "https://www.google.com/maps/search/?api=1&query=" + str(lat) + "," + str(lng)


def get_directions_url(lat, lng):
    return "https://www.google.com/maps/dir/?api=1&destination=" + str(lat) + "," + str(lng)


def get_maps_search_url(query):
    return "https://www.google.com/maps/search/" + query.replace(" ", "+")


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
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Note: This is AI-generated. Please consult a real doctor.", ln=True, align="C")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name


quotes = [
    "Health is wealth, take care of yourself.",
    "A healthy outside starts from the inside.",
    "Every day is a chance to get stronger.",
    "Take a deep breath, your health matters.",
    "Remember, self-care is not selfish."
]

# ------------------ Session State Init ------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = None
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = None
if 'show_doctors' not in st.session_state:
    st.session_state['show_doctors'] = False


# ==================== HEADER ====================
st.markdown("<div class='header-title'>🏥 AI Health ChatBot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your personal AI-powered health assistant</div>", unsafe_allow_html=True)
st.markdown("---")


# ==================== NOT LOGGED IN ====================
if not st.session_state['logged_in']:

    st.markdown("<h2 style='text-align:center; color:#00b4d8;'>Welcome! Please Login or Sign Up</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Role Selection
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='role-card'>
            <h1>👤</h1>
            <h2 style='color:#00b4d8;'>Patient</h2>
            <p style='color:#8892a4;'>Login or Sign up as a Patient to check your symptoms and book appointments</p>
        </div>
        """, unsafe_allow_html=True)
        role_patient = st.button("👤 Continue as Patient", use_container_width=True)

    with col2:
        st.markdown("""
        <div class='role-card'>
            <h1>👨‍⚕️</h1>
            <h2 style='color:#9b59b6;'>Doctor</h2>
            <p style='color:#8892a4;'>Login or Sign up as a Doctor to manage your appointments and patients</p>
        </div>
        """, unsafe_allow_html=True)
        role_doctor = st.button("👨‍⚕️ Continue as Doctor", use_container_width=True)

    if role_patient:
        st.session_state['selected_role'] = 'patient'
    if role_doctor:
        st.session_state['selected_role'] = 'doctor'

    # ------------------ PATIENT AUTH ------------------
    if st.session_state.get('selected_role') == 'patient':
        st.markdown("---")
        st.markdown("<h3 style='color:#00b4d8; text-align:center;'>👤 Patient Login / Sign Up</h3>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            p_email = st.text_input("Email", key="p_login_email")
            p_password = st.text_input("Password", type="password", key="p_login_pass")

            if st.button("🔐 Login as Patient", use_container_width=True):
                if not p_email or not p_password:
                    st.warning("Please fill all fields.")
                else:
                    user = login_patient(p_email, p_password)
                    if user:
                        st.session_state['logged_in'] = True
                        st.session_state['user_type'] = 'patient'
                        st.session_state['user_data'] = {
                            'id': user[0], 'name': user[1], 'email': user[2],
                            'age': user[4], 'gender': user[5], 'contact': user[6]
                        }
                        st.success("Welcome back, " + user[1] + "!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password!")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                p_name = st.text_input("Full Name", key="p_signup_name")
                p_email_s = st.text_input("Email", key="p_signup_email")
                p_password_s = st.text_input("Password", type="password", key="p_signup_pass")
            with col2:
                p_age = st.number_input("Age", min_value=1, max_value=120, value=25, key="p_signup_age")
                p_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="p_signup_gender")
                p_contact = st.text_input("Contact Number", key="p_signup_contact")

            if st.button("📝 Create Patient Account", use_container_width=True):
                if not p_name or not p_email_s or not p_password_s:
                    st.warning("Please fill all required fields.")
                else:
                    success, msg = signup_patient(p_name, p_email_s, p_password_s, p_age, p_gender, p_contact)
                    if success:
                        st.success(msg + " Please login now.")
                    else:
                        st.error(msg)

    # ------------------ DOCTOR AUTH ------------------
    elif st.session_state.get('selected_role') == 'doctor':
        st.markdown("---")
        st.markdown("<h3 style='color:#9b59b6; text-align:center;'>👨‍⚕️ Doctor Login / Sign Up</h3>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            d_email = st.text_input("Email", key="d_login_email")
            d_password = st.text_input("Password", type="password", key="d_login_pass")

            if st.button("🔐 Login as Doctor", use_container_width=True):
                if not d_email or not d_password:
                    st.warning("Please fill all fields.")
                else:
                    doctor = login_doctor(d_email, d_password)
                    if doctor:
                        st.session_state['logged_in'] = True
                        st.session_state['user_type'] = 'doctor'
                        st.session_state['user_data'] = {
                            'id': doctor[0], 'name': doctor[1], 'email': doctor[2],
                            'specialization': doctor[4], 'clinic': doctor[5],
                            'contact': doctor[6], 'slots': doctor[7],
                            'location': doctor[8], 'lat': doctor[9], 'lng': doctor[10]
                        }
                        st.success("Welcome Dr. " + doctor[1] + "!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password!")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                d_name = st.text_input("Doctor Name", key="d_signup_name")
                d_email_s = st.text_input("Email", key="d_signup_email")
                d_password_s = st.text_input("Password", type="password", key="d_signup_pass")
                d_spec = st.text_input("Specialization", key="d_signup_spec")
                d_clinic = st.text_input("Clinic/Hospital Name", key="d_signup_clinic")
            with col2:
                d_contact = st.text_input("Contact Number", key="d_signup_contact")
                d_slots = st.text_input("Available Slots (e.g. 6:00,6:30,7:00)", key="d_signup_slots")
                d_location = st.text_input("Location/Area", key="d_signup_location")
                d_lat = st.number_input("Latitude", value=17.3850, format="%.4f", key="d_signup_lat")
                d_lng = st.number_input("Longitude", value=78.4867, format="%.4f", key="d_signup_lng")

            if st.button("📝 Create Doctor Account", use_container_width=True):
                if not d_name or not d_email_s or not d_password_s or not d_spec:
                    st.warning("Please fill all required fields.")
                else:
                    success, msg = signup_doctor(d_name, d_email_s, d_password_s, d_spec, d_clinic, d_contact, d_slots, d_location, d_lat, d_lng)
                    if success:
                        st.success(msg + " Please login now.")
                    else:
                        st.error(msg)


# ==================== PATIENT DASHBOARD ====================
elif st.session_state['logged_in'] and st.session_state['user_type'] == 'patient':
    user = st.session_state['user_data']

    with st.sidebar:
        st.markdown("<h2 style='color:#00b4d8;'>🏥 HealthBot</h2>", unsafe_allow_html=True)
        st.markdown("<div class='welcome-card'><p>👤 <b>" + user['name'] + "</b></p><p style='color:#8892a4;'>Patient</p></div>", unsafe_allow_html=True)
        st.markdown("---")
        page = st.selectbox("Navigate To", [
            "🏠 Home & Diagnosis",
            "👨‍⚕️ Find Doctors",
            "🏥 Find Hospitals",
            "💊 Find Pharmacies",
            "📅 Book Appointment",
            "📋 My Appointments",
            "👤 My Profile"
        ])
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['user_type'] = None
            st.session_state['user_data'] = None
            st.session_state['show_doctors'] = False
            st.rerun()

    # ---- HOME & DIAGNOSIS ----
    if page == "🏠 Home & Diagnosis":
        st.markdown("<div class='section-title'>🤒 Describe Your Symptoms</div>", unsafe_allow_html=True)

        with st.expander("💡 Tips for better prediction"):
            st.write("✅ Add at least 4-5 symptoms for accurate results")
            st.write("✅ Use 'and' between symptoms")
            st.write("✅ Examples: fever, headache, nausea, joint pain, fatigue")

        symptoms_input = st.text_area("Type your symptoms below",
            placeholder="e.g. I have fever and headache and nausea and joint pain and fatigue", height=100)

        col1, col2 = st.columns(2)
        with col1:
            num_days = st.slider("Days with symptoms", 1, 30, 3)
        with col2:
            severity = st.slider("Severity (1=mild, 10=severe)", 1, 10, 5)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 Predict Disease"):
            if not symptoms_input:
                st.warning("Please describe your symptoms.")
            else:
                with st.spinner("Analyzing your symptoms..."):
                    symptoms_list = extract_symptoms(symptoms_input, cols)

                if not symptoms_list:
                    st.error("Could not detect symptoms. Please try: fever, headache, nausea, fatigue")
                else:
                    st.success("Detected " + str(len(symptoms_list)) + " symptoms: " + ", ".join(symptoms_list))
                    top3 = predict_top3(symptoms_list)
                    top_disease = top3[0]['disease']
                    top_confidence = top3[0]['confidence']

                    st.session_state['disease'] = top_disease
                    st.session_state['patient_name'] = user['name']

                    st.markdown("<div class='section-title'>🏆 Top 3 Possible Diseases</div>", unsafe_allow_html=True)

                    st.markdown(
                        "<div class='top1-card'><h3>🥇 #1 Most Likely: " + top3[0]['disease'] + "</h3>"
                        "<p>Confidence: <b style='color:#2ecc71; font-size:24px;'>" + str(top3[0]['confidence']) + "%</b></p>"
                        "<div style='background:#2ecc71; width:" + str(min(top3[0]['confidence'], 100)) + "%; height:12px; border-radius:6px;'></div>"
                        "</div>", unsafe_allow_html=True)

                    if top3[1]['confidence'] > 0:
                        st.markdown(
                            "<div class='top2-card'><h3>🥈 #2 Possible: " + top3[1]['disease'] + "</h3>"
                            "<p>Confidence: <b style='color:#f39c12; font-size:20px;'>" + str(top3[1]['confidence']) + "%</b></p>"
                            "<div style='background:#f39c12; width:" + str(min(top3[1]['confidence'], 100)) + "%; height:12px; border-radius:6px;'></div>"
                            "</div>", unsafe_allow_html=True)

                    if top3[2]['confidence'] > 0:
                        st.markdown(
                            "<div class='top3-card'><h3>🥉 #3 Also Possible: " + top3[2]['disease'] + "</h3>"
                            "<p>Confidence: <b style='color:#9b59b6; font-size:18px;'>" + str(top3[2]['confidence']) + "%</b></p>"
                            "<div style='background:#9b59b6; width:" + str(min(top3[2]['confidence'], 100)) + "%; height:12px; border-radius:6px;'></div>"
                            "</div>", unsafe_allow_html=True)

                    desc = description_list.get(top_disease, "No description available.")
                    st.markdown(
                        "<div class='result-card'><h4>About " + top_disease + "</h4><p>" + desc + "</p></div>",
                        unsafe_allow_html=True)

                    if top_disease in precautionDictionary:
                        precs = precautionDictionary[top_disease]
                        st.markdown(
                            "<div class='card'><h4>Suggested Precautions</h4>"
                            "<p>1. " + precs[0] + "</p><p>2. " + precs[1] + "</p>"
                            "<p>3. " + precs[2] + "</p><p>4. " + precs[3] + "</p></div>",
                            unsafe_allow_html=True)

                    if severity >= 7 or num_days >= 7:
                        st.error("Symptoms seem serious! Please consult a doctor immediately!")
                    else:
                        st.success("Follow the precautions and monitor your symptoms.")

                    precs_for_pdf = precautionDictionary.get(top_disease, ["", "", "", ""])
                    pdf_path = generate_pdf(user['name'], user['age'], user['gender'], symptoms_list, top_disease, top_confidence, desc, precs_for_pdf)
                    with open(pdf_path, "rb") as f:
                        st.download_button(label="📄 Download Health Report (PDF)", data=f,
                            file_name=user['name'] + "_health_report.pdf", mime="application/pdf", use_container_width=True)

                    st.markdown("---")
                    st.markdown(
                        "<div class='chatbot-bubble'>🤖 <b>HealthBot:</b> Based on your symptoms, you most likely have <b>" + top_disease + "</b>. "
                        "Would you like me to help you find a doctor and book an appointment?</div>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes, Find a Doctor", use_container_width=True):
                            st.session_state['show_doctors'] = True
                    with col2:
                        if st.button("❌ No, Thank You", use_container_width=True):
                            st.session_state['show_doctors'] = False
                            st.markdown(
                                "<div class='result-card'><h3>Thank you, " + user['name'] + "! 🌟</h3>"
                                "<p>Take care and get well soon!</p><p>💡 " + random.choice(quotes) + "</p></div>",
                                unsafe_allow_html=True)

                    if st.session_state.get('show_doctors', False):
                        st.markdown("<div class='section-title'>👨‍⚕️ Recommended Doctors</div>", unsafe_allow_html=True)
                        for _, doc in doctors_df.iterrows():
                            maps_url = get_maps_url(doc['lat'], doc['lng'])
                            directions_url = get_directions_url(doc['lat'], doc['lng'])
                            call_url = "tel:" + str(doc['contact'])
                            st.markdown(
                                "<div class='doctor-card'><h3>👨‍⚕️ " + doc['name'] + "</h3>"
                                "<p>🩺 <b>Specialization:</b> " + doc['specialization'] + "</p>"
                                "<p>🏥 <b>Clinic:</b> " + doc['clinic'] + "</p>"
                                "<p>📞 <b>Contact:</b> " + str(doc['contact']) + "</p>"
                                "<p>🕐 <b>Slots:</b> " + doc['slots'] + "</p><br>"
                                "<a href='" + call_url + "' class='call-btn'>📞 Call</a>"
                                "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 Maps</a>"
                                "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Directions</a>"
                                "</div>", unsafe_allow_html=True)
                        st.info("Go to Book Appointment from the sidebar to book a slot!")

    # ---- FIND DOCTORS ----
    elif page == "👨‍⚕️ Find Doctors":
        st.markdown("<div class='section-title'>👨‍⚕️ Find Nearby Doctors</div>", unsafe_allow_html=True)
        user_location = st.text_input("Enter your area/city", placeholder="e.g. Hyderabad, Ameerpet")
        if user_location:
            maps_search = get_maps_search_url("doctors in " + user_location)
            st.markdown("<a href='" + maps_search + "' target='_blank' class='map-btn'>🗺️ Search Doctors Near " + user_location + " on Google Maps</a><br><br>", unsafe_allow_html=True)

        specialization_filter = st.selectbox("Filter by Specialization", ["All"] + list(doctors_df['specialization'].unique()))
        filtered = doctors_df if specialization_filter == "All" else doctors_df[doctors_df['specialization'] == specialization_filter]

        for _, doc in filtered.iterrows():
            maps_url = get_maps_url(doc['lat'], doc['lng'])
            directions_url = get_directions_url(doc['lat'], doc['lng'])
            call_url = "tel:" + str(doc['contact'])
            st.markdown(
                "<div class='doctor-card'><h3>👨‍⚕️ " + doc['name'] + "</h3>"
                "<p>🩺 <b>Specialization:</b> " + doc['specialization'] + "</p>"
                "<p>🏥 <b>Clinic:</b> " + doc['clinic'] + "</p>"
                "<p>📞 <b>Contact:</b> " + str(doc['contact']) + "</p>"
                "<p>📍 <b>Location:</b> " + doc['location'] + "</p>"
                "<p>🕐 <b>Available Slots:</b> " + doc['slots'] + "</p><br>"
                "<a href='" + call_url + "' class='call-btn'>📞 Call Doctor</a>"
                "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 View on Maps</a>"
                "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions</a>"
                "</div>", unsafe_allow_html=True)

    # ---- FIND HOSPITALS ----
    elif page == "🏥 Find Hospitals":
        st.markdown("<div class='section-title'>🏥 Nearby Hospitals</div>", unsafe_allow_html=True)
        user_location = st.text_input("Enter your area/city", placeholder="e.g. Hyderabad, Ameerpet")
        if user_location:
            maps_search = get_maps_search_url("hospitals in " + user_location)
            st.markdown("<a href='" + maps_search + "' target='_blank' class='map-btn'>🗺️ Search Hospitals Near " + user_location + " on Google Maps</a><br><br>", unsafe_allow_html=True)

        hospitals = [
            {"name": "Apollo Hospital", "address": "Jubilee Hills, Hyderabad", "contact": "9040000000", "speciality": "Multi-Specialty", "beds": 500, "lat": 17.4239, "lng": 78.4738},
            {"name": "KIMS Hospital", "address": "Secunderabad, Hyderabad", "contact": "9040000001", "speciality": "Multi-Specialty", "beds": 400, "lat": 17.4399, "lng": 78.4983},
            {"name": "Yashoda Hospital", "address": "Somajiguda, Hyderabad", "contact": "9040000002", "speciality": "Multi-Specialty", "beds": 350, "lat": 17.4284, "lng": 78.4588},
            {"name": "Care Hospital", "address": "Banjara Hills, Hyderabad", "contact": "9040000003", "speciality": "Cardiac Care", "beds": 300, "lat": 17.4156, "lng": 78.4347},
            {"name": "Sunshine Hospital", "address": "PG Road, Hyderabad", "contact": "9040000004", "speciality": "General", "beds": 200, "lat": 17.4350, "lng": 78.4600},
        ]
        for hospital in hospitals:
            maps_url = get_maps_url(hospital['lat'], hospital['lng'])
            directions_url = get_directions_url(hospital['lat'], hospital['lng'])
            call_url = "tel:" + str(hospital['contact'])
            st.markdown(
                "<div class='hospital-card'><h3>🏥 " + hospital['name'] + "</h3>"
                "<p>📍 <b>Address:</b> " + hospital['address'] + "</p>"
                "<p>🩺 <b>Speciality:</b> " + hospital['speciality'] + "</p>"
                "<p>🛏️ <b>Beds:</b> " + str(hospital['beds']) + "</p>"
                "<p>📞 <b>Contact:</b> " + str(hospital['contact']) + "</p><br>"
                "<a href='" + call_url + "' class='call-btn'>📞 Call Hospital</a>"
                "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 View on Maps</a>"
                "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions</a>"
                "</div>", unsafe_allow_html=True)

    # ---- FIND PHARMACIES ----
    elif page == "💊 Find Pharmacies":
        st.markdown("<div class='section-title'>💊 Nearby Pharmacies</div>", unsafe_allow_html=True)
        user_location = st.text_input("Enter your area/city", placeholder="e.g. Hyderabad, Ameerpet")
        if user_location:
            maps_search = get_maps_search_url("pharmacy in " + user_location)
            st.markdown("<a href='" + maps_search + "' target='_blank' class='map-btn'>🗺️ Search Pharmacies Near " + user_location + " on Google Maps</a><br><br>", unsafe_allow_html=True)

        for _, pharmacy in pharmacies_df.iterrows():
            maps_url = get_maps_url(pharmacy['lat'], pharmacy['lng'])
            directions_url = get_directions_url(pharmacy['lat'], pharmacy['lng'])
            call_url = "tel:" + str(pharmacy['contact'])
            st.markdown(
                "<div class='pharmacy-card'><h3>💊 " + pharmacy['name'] + "</h3>"
                "<p>📍 <b>Address:</b> " + pharmacy['address'] + "</p>"
                "<p>📞 <b>Contact:</b> " + str(pharmacy['contact']) + "</p><br>"
                "<a href='" + call_url + "' class='call-btn'>📞 Call Pharmacy</a>"
                "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 View on Maps</a>"
                "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions</a>"
                "</div>", unsafe_allow_html=True)

    # ---- BOOK APPOINTMENT ----
    elif page == "📅 Book Appointment":
        st.markdown("<div class='section-title'>📅 Book Doctor Appointment</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Your Name", value=user['name'])
        with col2:
            patient_contact = st.text_input("Your Contact", value=user['contact'])

        disease_info = st.text_input("Disease/Reason", value=st.session_state.get('disease', ''))
        selected_doctor = st.selectbox("Select Doctor", doctors_df['name'].tolist())
        doctor_info = doctors_df[doctors_df['name'] == selected_doctor].iloc[0]

        maps_url = get_maps_url(doctor_info['lat'], doctor_info['lng'])
        directions_url = get_directions_url(doctor_info['lat'], doctor_info['lng'])
        call_url = "tel:" + str(doctor_info['contact'])

        st.markdown(
            "<div class='doctor-card'>"
            "<p>🏥 <b>Clinic:</b> " + doctor_info['clinic'] + "</p>"
            "<p>🩺 <b>Specialization:</b> " + doctor_info['specialization'] + "</p>"
            "<p>📍 <b>Location:</b> " + doctor_info['location'] + "</p>"
            "<p>📞 <b>Contact:</b> " + str(doctor_info['contact']) + "</p><br>"
            "<a href='" + call_url + "' class='call-btn'>📞 Call Doctor</a>"
            "<a href='" + maps_url + "' target='_blank' class='map-btn'>📍 View on Maps</a>"
            "<a href='" + directions_url + "' target='_blank' class='map-btn'>🗺️ Get Directions</a>"
            "</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            selected_slot = st.selectbox("Select Time Slot", doctor_info['slots'].split(','))
        with col2:
            appointment_date = st.date_input("Select Date")

        if st.button("Confirm Appointment"):
            conn = sqlite3.connect('health_app.db')
            c = conn.cursor()
            c.execute('''INSERT INTO appointments (patient_name, patient_contact, patient_email, doctor_name, clinic, date, slot, disease)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (patient_name, patient_contact, user['email'], selected_doctor, doctor_info['clinic'], str(appointment_date), selected_slot, disease_info))
            conn.commit()
            conn.close()
            st.balloons()
            st.markdown(
                "<div class='result-card'><h2>🎉 Appointment Confirmed!</h2>"
                "<p><b>Patient:</b> " + patient_name + "</p>"
                "<p><b>Doctor:</b> " + selected_doctor + "</p>"
                "<p><b>Clinic:</b> " + doctor_info['clinic'] + "</p>"
                "<p><b>Date:</b> " + str(appointment_date) + "</p>"
                "<p><b>Time:</b> " + selected_slot + "</p>"
                "<p><b>Reason:</b> " + disease_info + "</p></div>", unsafe_allow_html=True)

    # ---- MY APPOINTMENTS ----
    elif page == "📋 My Appointments":
        st.markdown("<div class='section-title'>📋 My Appointments</div>", unsafe_allow_html=True)
        conn = sqlite3.connect('health_app.db')
        df = pd.read_sql_query("SELECT * FROM appointments WHERE patient_email=?", conn, params=(user['email'],))
        conn.close()
        if df.empty:
            st.info("You have no appointments yet. Book one from the Book Appointment page!")
        else:
            st.success("You have " + str(len(df)) + " appointment(s)!")
            for _, apt in df.iterrows():
                st.markdown(
                    "<div class='result-card'><h4>📅 Appointment #" + str(apt['id']) + "</h4>"
                    "<p><b>Doctor:</b> " + str(apt['doctor_name']) + "</p>"
                    "<p><b>Clinic:</b> " + str(apt['clinic']) + "</p>"
                    "<p><b>Date:</b> " + str(apt['date']) + "</p>"
                    "<p><b>Time:</b> " + str(apt['slot']) + "</p>"
                    "<p><b>Reason:</b> " + str(apt['disease']) + "</p></div>", unsafe_allow_html=True)

    # ---- MY PROFILE ----
    elif page == "👤 My Profile":
        st.markdown("<div class='section-title'>👤 My Profile</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='result-card'>"
            "<h3>👤 " + user['name'] + "</h3>"
            "<p>📧 <b>Email:</b> " + user['email'] + "</p>"
            "<p>🎂 <b>Age:</b> " + str(user['age']) + "</p>"
            "<p>⚧ <b>Gender:</b> " + str(user['gender']) + "</p>"
            "<p>📞 <b>Contact:</b> " + str(user['contact']) + "</p>"
            "</div>", unsafe_allow_html=True)


# ==================== DOCTOR DASHBOARD ====================
elif st.session_state['logged_in'] and st.session_state['user_type'] == 'doctor':
    doctor = st.session_state['user_data']

    with st.sidebar:
        st.markdown("<h2 style='color:#9b59b6;'>🏥 HealthBot</h2>", unsafe_allow_html=True)
        st.markdown("<div class='welcome-card'><p>👨‍⚕️ <b>Dr. " + doctor['name'] + "</b></p><p style='color:#8892a4;'>" + doctor['specialization'] + "</p></div>", unsafe_allow_html=True)
        st.markdown("---")
        page = st.selectbox("Navigate To", [
            "🏠 Doctor Dashboard",
            "📋 My Appointments",
            "👨‍⚕️ My Profile"
        ])
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['user_type'] = None
            st.session_state['user_data'] = None
            st.rerun()

    # ---- DOCTOR DASHBOARD ----
    if page == "🏠 Doctor Dashboard":
        st.markdown("<div class='section-title'>🏠 Welcome Dr. " + doctor['name'] + "!</div>", unsafe_allow_html=True)

        # Stats
        conn = sqlite3.connect('health_app.db')
        total_appointments = pd.read_sql_query("SELECT COUNT(*) as count FROM appointments WHERE doctor_name=?",
            conn, params=(doctor['name'],)).iloc[0]['count']
        today_appointments = pd.read_sql_query("SELECT COUNT(*) as count FROM appointments WHERE doctor_name=? AND date=date('now')",
            conn, params=(doctor['name'],)).iloc[0]['count']
        conn.close()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                "<div class='metric-card'><div class='metric-value'>" + str(total_appointments) + "</div>"
                "<div class='metric-label'>Total Appointments</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                "<div class='metric-card'><div class='metric-value'>" + str(today_appointments) + "</div>"
                "<div class='metric-label'>Today's Appointments</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(
                "<div class='metric-card'><div class='metric-value'>" + doctor['specialization'] + "</div>"
                "<div class='metric-label'>Specialization</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>📋 Recent Appointments</div>", unsafe_allow_html=True)
        conn = sqlite3.connect('health_app.db')
        df = pd.read_sql_query("SELECT * FROM appointments WHERE doctor_name=? ORDER BY created_at DESC LIMIT 5",
            conn, params=(doctor['name'],))
        conn.close()

        if df.empty:
            st.info("No appointments yet.")
        else:
            for _, apt in df.iterrows():
                st.markdown(
                    "<div class='doctor-card'>"
                    "<h4>👤 Patient: " + str(apt['patient_name']) + "</h4>"
                    "<p>📞 <b>Contact:</b> " + str(apt['patient_contact']) + "</p>"
                    "<p>📅 <b>Date:</b> " + str(apt['date']) + "</p>"
                    "<p>🕐 <b>Time:</b> " + str(apt['slot']) + "</p>"
                    "<p>🩺 <b>Reason:</b> " + str(apt['disease']) + "</p>"
                    "</div>", unsafe_allow_html=True)

    # ---- ALL APPOINTMENTS ----
    elif page == "📋 My Appointments":
        st.markdown("<div class='section-title'>📋 All My Appointments</div>", unsafe_allow_html=True)
        conn = sqlite3.connect('health_app.db')
        df = pd.read_sql_query("SELECT * FROM appointments WHERE doctor_name=? ORDER BY date DESC",
            conn, params=(doctor['name'],))
        conn.close()

        if df.empty:
            st.info("No appointments yet.")
        else:
            st.success("Total " + str(len(df)) + " appointment(s)!")
            for _, apt in df.iterrows():
                st.markdown(
                    "<div class='doctor-card'>"
                    "<h4>👤 " + str(apt['patient_name']) + "</h4>"
                    "<p>📞 <b>Contact:</b> " + str(apt['patient_contact']) + "</p>"
                    "<p>📧 <b>Email:</b> " + str(apt['patient_email']) + "</p>"
                    "<p>📅 <b>Date:</b> " + str(apt['date']) + "</p>"
                    "<p>🕐 <b>Time:</b> " + str(apt['slot']) + "</p>"
                    "<p>🩺 <b>Reason:</b> " + str(apt['disease']) + "</p>"
                    "</div>", unsafe_allow_html=True)

    # ---- DOCTOR PROFILE ----
    elif page == "👨‍⚕️ My Profile":
        st.markdown("<div class='section-title'>👨‍⚕️ My Profile</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='result-card'>"
            "<h3>👨‍⚕️ Dr. " + doctor['name'] + "</h3>"
            "<p>📧 <b>Email:</b> " + doctor['email'] + "</p>"
            "<p>🩺 <b>Specialization:</b> " + doctor['specialization'] + "</p>"
            "<p>🏥 <b>Clinic:</b> " + str(doctor['clinic']) + "</p>"
            "<p>📞 <b>Contact:</b> " + str(doctor['contact']) + "</p>"
            "<p>🕐 <b>Slots:</b> " + str(doctor['slots']) + "</p>"
            "<p>📍 <b>Location:</b> " + str(doctor['location']) + "</p>"
            "</div>", unsafe_allow_html=True)
