import streamlit as st
import pickle
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load ML model
model = pickle.load(open("dt_crop.pkl", "rb"))

# Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Page config
st.set_page_config(
    page_title="AI Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 AI-Based Crop Recommendation System")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🌱 Soil & Climate Inputs")

N = st.sidebar.number_input("Nitrogen (N)", 0, 140)
P = st.sidebar.number_input("Phosphorus (P)", 0, 145)
K = st.sidebar.number_input("Potassium (K)", 0, 205)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0)
ph = st.sidebar.number_input("pH Value", 0.0, 14.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 300.0)

# -------------------------------
# Crop Recommendation
# -------------------------------
if st.sidebar.button("🌱 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0].upper()}**")

# -------------------------------
# Chatbot Section
# -------------------------------
st.divider()
st.header("🤖 Farmer Assistant Chatbot (Groq AI)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Chat input
user_input = st.chat_input("Ask about crops, fertilizers, irrigation, weather...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # AI response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agricultural assistant. "
                    "Give clear, simple, and farmer-friendly answers."
                )
            }
        ] + st.session_state.chat_history,
    )

    bot_reply = response.choices[0].message.content

    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": bot_reply}
    )

