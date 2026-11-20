# Fichier de Nadine Zerguine : client de l'application (Streamlit).
# Il n'a aucun modèle : il envoie l'image au serveur (serveur.py) et affiche la réponse.
import requests
import streamlit as st

URL_SERVEUR = "http://127.0.0.1:5000"

st.title("Classification de fleurs")
modele = st.radio("Choisis un modèle", ["MLP", "Linear Model", "SVM", "RBF"], horizontal=True)
fichier = st.file_uploader("Choisis une photo de fleur", type=["jpg", "jpeg", "png"])

if fichier is not None:
    st.image(fichier, width=300)

    if st.button("Classifier"):
        try:
            reponse = requests.post(URL_SERVEUR + "/predict",
                                    data={"modele": modele},
                                    files={"image": fichier.getvalue()})
            resultat = reponse.json()
        except requests.exceptions.ConnectionError:
            st.error("Le serveur ne répond pas : lance d'abord  python app/serveur.py")
            st.stop()

        if "erreur" in resultat:
            st.error(resultat["erreur"])
        else:
            st.success("Prédiction : " + resultat["classe"])
            st.write("Scores du modèle (ce ne sont pas des probabilités) :")
            for nom, score in zip(resultat["classes"], resultat["scores"]):
                st.write(f"{nom} : {score:.4f}")
