
"""
Page 3 : Interface de dictée vocale avec l'utilisation du modele Whisper-tiny de OpenAI
"""

import os
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Dictée Vocale | Brief 2",
    page_icon="🎙️",
    layout="wide",
)

st.title("Page 3 : Dictée Vocale")
st.markdown("""
Cette page permet de dicter du texte à l'aide du micro et utilise le modèle **Whisper-tiny** de OpenAI pour la transcription automatique.
""")

audio_enregistre = st.audio_input(
    "Enregistrez votre dictée vocale ici :",sample_rate=16000
)

if audio_enregistre is not None:
    st.audio(audio_enregistre)
    st.success("Audio enregistré avec succès.")
    try:
        with st.spinner("Transcription en cours..."):
            fichiers = {"file": (
                            audio_enregistre.name, 
                            audio_enregistre.getvalue(), 
                            audio_enregistre.type
                            )
                        }
            response = requests.post(f"{FASTAPI_URL}/audio/transcribe", files=fichiers, timeout=120)
            if response.status_code == 200:
                result = response.json()

                st.write("Fichier :", result["filename"])
                st.write("Transcription :", result["transcription"])
                st.write("Langue :", result["language"])
                st.write("Temps d'exécution :", result["execution_time_seconds"])

            else:
                st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")

    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
