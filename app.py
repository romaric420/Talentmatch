"""
TalentMatch - Scoring de Candidatures RH
Page d'accueil
"""

import streamlit as st

st.set_page_config(
    page_title="TalentMatch",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #9B59B6;
        margin-bottom: 1rem;
    }
    .tech-badge {
        background: #9B59B6;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">👥 TalentMatch</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Système Intelligent de Scoring et Matching de Candidats</p>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("À propos du projet")
    
    st.markdown("""
    **TalentMatch** est une application de Machine Learning conçue pour optimiser 
    le processus de recrutement. Elle utilise des algorithmes de scoring et de 
    matching pour identifier les meilleurs candidats.
    
    ### Problématique Business
    
    Le recrutement est un processus coûteux et chronophage :
    
    - Tri manuel de centaines de CV
    - Difficulté à comparer objectivement les candidats
    - Risque de passer à côté de bons profils
    - Besoin de standardiser l'évaluation
    """)
    
    st.header("Fonctionnalités")
    
    st.markdown("""
    <div class="feature-card">
        <strong>Dashboard</strong><br>
        Vue d'ensemble du vivier de candidats et KPIs de recrutement
    </div>
    
    <div class="feature-card">
        <strong>Recherche</strong><br>
        Filtrage multi-critères avec scoring en temps réel
    </div>
    
    <div class="feature-card">
        <strong>Matching</strong><br>
        Identification de profils similaires avec KNN
    </div>
    
    <div class="feature-card">
        <strong>Analyse</strong><br>
        Segmentation et clustering des talents
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.header("Technologies")
    
    st.markdown("""
    <span class="tech-badge">Python</span>
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">Scikit-learn</span>
    <span class="tech-badge">Pandas</span>
    <span class="tech-badge">Plotly</span>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("Algorithmes")
    
    st.markdown("""
    | Algorithme | Usage |
    |------------|-------|
    | KNN | Matching de profils |
    | Scoring | Évaluation multi-critères |
    | PCA | Réduction de dimension |
    """)
    
    st.markdown("---")
    
    st.header("Critères de Scoring")
    
    st.markdown("""
    - Compétences techniques
    - Expérience professionnelle
    - Soft skills
    - Évaluations (entretiens)
    """)

st.markdown("---")

st.header("Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**1 - Dashboard**\n\nVue d'ensemble")

with col2:
    st.info("**2 - Recherche**\n\nFiltrage et scoring")

with col3:
    st.info("**3 - Matching**\n\nProfils similaires")

with col4:
    st.info("**4 - Analyse**\n\nSegmentation")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Utilisez le menu latéral pour naviguer"
    "</div>",
    unsafe_allow_html=True
)


# Footer 
st.markdown("---")
st.markdown(f"""
<div style="
    background-color: #6F4F28; 
    padding: 20px; 
    border-radius: 10px; 
    text-align: center;
">
    <p style="color: white; margin: 5px;">TALENTMATCH Dashboard</p>
    <p style="color: #ccc; font-size: 12px; margin: 5px;">© 2026 - Romaric</p>
</div>
""", unsafe_allow_html=True)