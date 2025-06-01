import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ⚙️ Configuration initiale
st.set_page_config(page_title="Diagnostic Cancer du Sein", page_icon="🩺", layout="centered")

# 🎯 Chargement des ressources
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

@st.cache_data
def load_data():
    df = pd.read_csv("wisc_bc_data.csv")
    df.drop(['id'], axis=1, errors='ignore', inplace=True)
    return df

with st.spinner("📥 Chargement des données..."):
    df = load_data()

# 🧾 En-tête
st.markdown("<h1 style='text-align: center;'>🩺 Diagnostic du Cancer du Sein</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>Cette application prédit si une tumeur est <strong style='color:blue;'>bénigne</strong> ou <strong style='color:red;'>maligne</strong> à partir des caractéristiques *_worst.</p>
""", unsafe_allow_html=True)

# 📁 Affichage optionnel des données
with st.expander("📂 Afficher le dataset complet"):
    st.dataframe(df.style.set_properties(**{'text-align': 'left'}), use_container_width=True)

# 🎛️ Fonction d'entrée utilisateur
def user_input_features(enabled=True):
    st.markdown("### 🧮 Paramètres à saisir dans la barre latérale")
    worst_features = [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    features = {}
    for feature in worst_features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        label = feature.replace("_worst", "").capitalize()
        features[feature] = st.slider(label=f"{label} (worst)",min_value=min_val,max_value=max_val,value=mean_val,
                                      step=(max_val - min_val) / 100,disabled=not enabled)
    return pd.DataFrame([features])

# 📥 Formulaire de prédiction dans la sidebar
with st.sidebar.form("prediction_form"):
    st.header("🎛️ Caractéristiques")
    input_df = user_input_features(enabled=True)
    submit_button = st.form_submit_button("🔍 Faire la prédiction")

# 🧾 Affichage des valeurs saisies
st.markdown("### 📄 Données saisies")
st.dataframe(input_df, use_container_width=True)

# 🔍 Si bouton cliqué
if submit_button:
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)

    # 🧪 Affichage résultat
    result_color = "#d1f0d1" if prediction[0] == 0 else "#f8d7da"
    diagnosis = "🔵 Bénin" if prediction[0] == 0 else "🔴 Malin"

    st.markdown(f"""
    <div style='background-color: {result_color}; padding: 20px; border-radius: 10px; border: 1px solid #ccc;'>
        <h3 style='text-align: center;'>🧬 Résultat de la prédiction</h3>
        <p style='font-size: 24px; text-align: center;'><strong>{diagnosis}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Probabilités
    st.markdown("### 📊 Probabilités de la prédiction")
    st.write(f"**Bénin :** {prediction_proba[0][0]:.2%}")
    st.write(f"**Malin :** {prediction_proba[0][1]:.2%}")

    # Graphique matplotlib
    fig, ax = plt.subplots()
    labels = ['Bénin', 'Malin']
    colors = ['skyblue', 'salmon']
    ax.bar(labels, prediction_proba[0], color=colors)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probabilité")
    ax.set_title("Probabilité du diagnostic")
    st.pyplot(fig)
else:
    st.info("➡️ Utilisez la barre latérale pour saisir les données, puis cliquez sur **Faire la prédiction**.")

# Pied de page personnalisé
st.markdown(
    """<hr style='margin-top: 30px;'>
    <div style='text-align: right; font-size: 0.8rem; color: gray; opacity: 0.6;'>
        📜 Copyright - MBOG-Aime-Thierry
    </div>
    """, unsafe_allow_html=True
)
