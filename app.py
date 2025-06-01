import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="WISC BC - Analyse et Interprétation", layout="wide")

st.title("🔬 Analyse avancée du dataset WISC BC")
st.markdown("""
    Cette application interactive vous permet d'explorer en profondeur le dataset du cancer du sein du Wisconsin.
    Vous pouvez comparer les performances de différents modèles, visualiser les distributions, et effectuer des prédictions manuelles.
""")

# --- Fonctions de chargement et d'entraînement ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("wisc_bc_data.csv")
        return df
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'wisc_bc_data.csv' est introuvable.")
        st.stop()

@st.cache_resource
def train_and_evaluate_models(X_train_df, y_train_series, X_test_df, y_test_series):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = []
    fitted_models = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train_df, y_train_series)
        y_pred = model.predict(X_test_df)
        y_proba = model.predict_proba(X_test_df)[:, 1]

        acc = accuracy_score(y_test_series, y_pred)
        roc = roc_auc_score(y_test_series, y_proba)

        results.append({"Modèle": name, "Accuracy": acc, "ROC AUC": roc})
        fitted_models[name] = model
        predictions[name] = (y_pred, y_proba)

    return pd.DataFrame(results), fitted_models, predictions

# --- Préparation des données ---
df = load_data()
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"].map({"M": 1, "B": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

results_df, fitted_models, predictions = train_and_evaluate_models(X_train_df, y_train_series, X_test_df, y_test_series)

best_model_name = results_df.loc[results_df["ROC AUC"].idxmax(), "Modèle"]
best_model = fitted_models[best_model_name]

# --- Interface à onglets ---
tabs = st.tabs([
    "📊 Comparaison des modèles",
    "📈 Performance du meilleur modèle",
    "🔎 Distribution des variables",
    "🔮 Prédiction manuelle"
])

# --- Onglet 1: Comparaison des modèles ---
with tabs[0]:
    st.header("📊 Comparaison des performances des modèles")
    st.dataframe(results_df.style.highlight_max(subset=["Accuracy", "ROC AUC"], color="lightgreen").format({"Accuracy": "{:.3f}", "ROC AUC": "{:.3f}"}))
    st.success(f"Modèle le plus performant (ROC AUC) : **{best_model_name}**")

# --- Onglet 2: Performance du meilleur modèle ---
with tabs[1]:
    st.header(f"📈 Performance du modèle : **{best_model_name}**")
    y_pred, y_proba = predictions[best_model_name]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test_series, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=["Bénin Prédit", "Malin Prédit"],
                    yticklabels=["Bénin Réel", "Malin Réel"])
        ax_cm.set_title("Matrice de confusion")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col2:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test_series, y_proba)
        roc_auc = roc_auc_score(y_test_series, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='darkorange', lw=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlabel("Faux positifs")
        ax_roc.set_ylabel("Vrais positifs")
        ax_roc.set_title("Courbe ROC")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        plt.close(fig_roc)

# --- Onglet 3: Distribution des variables ---
with tabs[2]:
    st.header("🔎 Distribution des variables")
    feat = st.selectbox("Sélectionner une caractéristique :", X.columns)
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=feat, hue="diagnosis", multiple="stack", palette="Set2", ax=ax_dist, kde=True)
    ax_dist.set_title(f"Distribution de '{feat}' selon le diagnostic")
    st.pyplot(fig_dist)
    plt.close(fig_dist)

# --- Onglet 4: Prédiction manuelle ---
with tabs[3]:
    st.header("🔮 Prédiction manuelle d'un cas")
    st.markdown("Entrez les valeurs des caractéristiques pour simuler une prédiction. Modèle utilisé : **{}**.".format(best_model_name))

    with st.form("manual_prediction_form"):
        cols = st.columns(3)
        input_data = {}
        for i, col in enumerate(cols):
            for feat in X.columns[i::3]:
                input_data[feat] = col.number_input(feat, min_value=0.0, format="%.3f")

        submitted = st.form_submit_button("🔍 Lancer la prédiction")

    if submitted:
        input_df = pd.DataFrame([input_data])
        pred_label = best_model.predict(input_df)[0]
        pred_proba = best_model.predict_proba(input_df)[0][1]
        result_text = "🧬 Résultat : **Malin (M)**" if pred_label == 1 else "🧬 Résultat : **Bénin (B)**"

        st.success(result_text)
        st.metric(label="Probabilité d'être Malin", value=f"{pred_proba:.2%}")
        st.markdown("### 🧾 Détail des valeurs saisies")
        st.dataframe(input_df.T.rename(columns={0: "Valeur entrée"}))

# --- Export des résultats ---
st.markdown("---")
st.subheader("💾 Exporter les résultats")
col1, col2 = st.columns(2)

with col1:
    csv_results = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger comparaison (CSV)", csv_results, "comparaison_modeles.csv", "text/csv")

with col2:
    pred_test_df = pd.DataFrame({
        "y_true": y_test_series,
        "y_pred": predictions[best_model_name][0],
        "y_proba": predictions[best_model_name][1]
    })
    csv_pred = pred_test_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger prédictions test", csv_pred, f"predictions_{best_model_name}.csv", "text/csv")

st.markdown("---")
st.caption("Application développée avec Streamlit, Scikit-learn, XGBoost, Matplotlib et Seaborn.")
