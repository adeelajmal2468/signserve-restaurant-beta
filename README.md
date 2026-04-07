# signserve-restaurant-beta
A polished Streamlit web app for real-time sign-to-text recognition in restaurant environments using TensorFlow, MediaPipe, and browser-based webcam streaming.

# 🍽️ SignServe Beta

A polished **Streamlit web app** for **real-time sign-to-text recognition** in restaurant environments.

SignServe Beta is designed to help restaurants test a browser-based experience where users can perform supported signs in front of a webcam and receive live recognition results directly in the app.

---

## ✨ Overview

SignServe Beta combines:

- **TensorFlow** for sign classification
- **MediaPipe Holistic** for landmark extraction
- **Streamlit** for a clean, modern web interface
- **WebRTC** for live browser-based webcam streaming

The app is built for **beta testing in real restaurant settings**, making it easy to share a link with testers and gather early feedback on usability, recognition quality, and overall experience.

---

## 🎯 Key Features

- 📷 **Live webcam-based sign recognition**
- 🧠 **Real-time prediction using a trained TensorFlow model**
- ✋ **Idle protection** with **“No sign recognized”** when no meaningful sign is detected
- 🎨 **Modern and attractive UI** designed for demos and beta testing
- 🪟 **Browser-based experience** — no OpenCV desktop popup windows for testers
- 📊 **Prediction confidence and status display**
- 🕘 **Recognition history panel**
- 🚀 **Ready for deployment on Streamlit Community Cloud**

---

## 🖼️ Use Case

This project is especially useful for:

- restaurant accessibility demos
- sign-to-text interaction prototypes
- beta testing with real users
- final year project showcases
- applied AI + HCI demonstrations

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **streamlit-webrtc**
- **TensorFlow / Keras**
- **MediaPipe**
- **OpenCV**
- **NumPy**

---

## 📁 Project Structure

```text
signserve-beta/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── models/
    ├── best_bigru_attention_aug.keras
    └── class_names.json
