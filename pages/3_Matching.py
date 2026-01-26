"""
Matching KNN - TalentMatch
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Matching - TalentMatch", layout="wide")


# GÉNÉRATION DES DONNÉES


TECH_SKILLS = ['Python', 'SQL', 'Excel', 'Tableau', 'Power BI', 'Machine Learning', 'Statistics', 'R']
SOFT_SKILLS = ['Communication', 'Leadership', 'Travail équipe', 'Résolution problèmes', 'Créativité', 'Adaptabilité']

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    
    domains = ['Finance', 'Marketing', 'Santé', 'E-commerce', 'Industrie', 'Tech']
    degrees = ['Bac+3', 'Bac+5 École', 'Bac+5 Université', 'Doctorat', 'Formation Pro']
    positions = ['Data Analyst Junior', 'Data Analyst', 'Data Analyst Senior', 'Business Analyst', 'Data Scientist Junior', 'Consultant BI']
    
    data = {
        'candidate_id': [f'CAND_{i:04d}' for i in range(n)],
        'name': [f'Candidat {i}' for i in range(n)],
        'age': np.random.normal(30, 6, n).clip(22, 55).astype(int),
        'experience_years': np.random.exponential(4, n).clip(0, 20).round(1),
        'degree': np.random.choice(degrees, n, p=[0.15, 0.35, 0.30, 0.05, 0.15]),
        'domain_expertise': np.random.choice(domains, n),
        'current_position': np.random.choice(positions, n),
        'expected_salary': np.zeros(n),
        'availability_weeks': np.random.choice([0, 1, 2, 4, 8, 12], n, p=[0.1, 0.25, 0.25, 0.2, 0.15, 0.05]),
        'interview_score': np.random.normal(70, 15, n).clip(30, 100).round(0).astype(int),
        'technical_test_score': np.random.normal(65, 18, n).clip(20, 100).round(0).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    for skill in TECH_SKILLS:
        base = np.random.normal(50, 25, n)
        exp_bonus = df['experience_years'] * 2
        df[f'skill_{skill}'] = (base + exp_bonus).clip(0, 100).round(0).astype(int)
    
    for skill in SOFT_SKILLS:
        df[f'soft_{skill}'] = np.random.normal(65, 20, n).clip(0, 100).round(0).astype(int)
    
    df['expected_salary'] = (
        32000 + 
        df['experience_years'] * 3500 +
        (df['degree'] == 'Bac+5 École').astype(int) * 8000 +
        np.random.normal(0, 5000, n)
    ).clip(28000, 120000).astype(int)
    
    df['composite_score'] = (
        (df['skill_Python'] + df['skill_SQL'] + df['skill_Statistics']) / 3 * 0.4 +
        df['experience_years'] * 5 * 0.25 +
        (df['soft_Communication'] + df['soft_Travail équipe']) / 2 * 0.2 +
        (df['interview_score'] + df['technical_test_score']) / 2 * 0.15
    ).round(1)
    
    return df


# PAGE


st.title("Matching de Profils (KNN)")
st.markdown("Identification de candidats similaires avec l'algorithme K-Nearest Neighbors")

df = load_data()

st.info("""
**Comment fonctionne le Matching KNN ?**

L'algorithme K-Nearest Neighbors identifie les K candidats les plus similaires 
à un profil de référence en calculant la distance euclidienne dans l'espace 
des caractéristiques sélectionnées.
""")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuration")
    
    reference_candidate = st.selectbox("Candidat de référence", df['candidate_id'].tolist())
    n_neighbors = st.slider("Nombre de profils similaires", 3, 15, 5)
    
    st.markdown("---")
    st.subheader("Critères de matching")
    
    use_tech = st.checkbox("Compétences techniques", value=True)
    use_soft = st.checkbox("Soft skills", value=False)
    use_exp = st.checkbox("Expérience", value=True)
    
    st.markdown("---")
    st.subheader("Profil de Référence")
    
    ref = df[df['candidate_id'] == reference_candidate].iloc[0]
    st.markdown(f"**{ref['candidate_id']}**")
    st.markdown(f"Poste: {ref['current_position']}")
    st.markdown(f"Expérience: {ref['experience_years']:.1f} ans")
    st.markdown(f"Score: {ref['composite_score']:.1f}/100")

with col2:
    # Construction des features
    feature_cols = []
    
    if use_exp:
        feature_cols.append('experience_years')
    
    if use_tech:
        feature_cols.extend([f'skill_{s}' for s in TECH_SKILLS[:6]])
    
    if use_soft:
        feature_cols.extend([f'soft_{s}' for s in SOFT_SKILLS])
    
    if len(feature_cols) > 0:
        X = df[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
        knn.fit(X_scaled)
        
        ref_idx = df[df['candidate_id'] == reference_candidate].index[0]
        ref_data = X_scaled[ref_idx].reshape(1, -1)
        
        distances, indices = knn.kneighbors(ref_data)
        
        # Exclusion du candidat lui-même
        similar_indices = [i for i in indices[0] if df.iloc[i]['candidate_id'] != reference_candidate][:n_neighbors]
        similar_distances = distances[0][1:n_neighbors+1]
        
        st.header("Profils Similaires")
        
        similar_df = df.iloc[similar_indices][
            ['candidate_id', 'current_position', 'experience_years', 'domain_expertise', 'composite_score', 'expected_salary']
        ].copy()
        
        similar_df['Similarité'] = [f"{max(0, 100 - d*10):.1f}%" for d in similar_distances]
        similar_df['Score'] = similar_df['composite_score'].apply(lambda x: f"{x:.1f}")
        similar_df['Exp.'] = similar_df['experience_years'].apply(lambda x: f"{x:.1f} ans")
        similar_df['Salaire'] = similar_df['expected_salary'].apply(lambda x: f"{x:,} €")
        
        st.dataframe(
            similar_df[['candidate_id', 'current_position', 'Exp.', 'domain_expertise', 'Score', 'Salaire', 'Similarité']],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Comparaison radar
        st.subheader("Comparaison des Compétences")
        
        comparison_skills = ['Python', 'SQL', 'Statistics', 'Excel', 'Machine Learning']
        
        fig = go.Figure()
        colors = ['#9B59B6', '#E94F37', '#2ECC71', '#F39C12', '#3498DB']
        
        # Candidat de référence
        ref_values = [ref[f'skill_{s}'] for s in comparison_skills]
        fig.add_trace(go.Scatterpolar(
            r=ref_values + [ref_values[0]],
            theta=comparison_skills + [comparison_skills[0]],
            name=f"{reference_candidate} (Référence)",
            line=dict(color=colors[0], width=3)
        ))
        
        # Candidats similaires (top 3)
        for i, idx in enumerate(similar_indices[:3]):
            cand = df.iloc[idx]
            values = [cand[f'skill_{s}'] for s in comparison_skills]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=comparison_skills + [comparison_skills[0]],
                name=cand['candidate_id'],
                line=dict(color=colors[i+1], width=2)
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Comparaison des Profils",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sélectionnez au moins un critère de matching.")
