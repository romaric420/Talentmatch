"""
Recherche de Candidats - TalentMatch
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Recherche - TalentMatch", layout="wide")


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


st.title("Recherche de Candidats")
st.markdown("Filtrage multi-critères et scoring en temps réel")

df = load_data()

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Critères de Recherche")
    
    search_domain = st.selectbox("Domaine", ['Tous'] + list(df['domain_expertise'].unique()))
    min_score = st.slider("Score minimum", 0, 100, 50)
    min_experience = st.slider("Expérience minimum (ans)", 0, 15, 0)
    max_salary = st.number_input("Budget salaire max (€)", 30000, 150000, 80000, 5000)
    
    required_skills = st.multiselect(
        "Compétences requises",
        TECH_SKILLS[:6],
        default=['Python', 'SQL']
    )
    
    min_skill_level = st.slider("Niveau minimum requis", 0, 100, 60)

with col2:
    # Filtrage
    df_search = df.copy()
    
    if search_domain != 'Tous':
        df_search = df_search[df_search['domain_expertise'] == search_domain]
    
    df_search = df_search[
        (df_search['composite_score'] >= min_score) &
        (df_search['experience_years'] >= min_experience) &
        (df_search['expected_salary'] <= max_salary)
    ]
    
    for skill in required_skills:
        col_name = f'skill_{skill}'
        if col_name in df_search.columns:
            df_search = df_search[df_search[col_name] >= min_skill_level]
    
    st.header(f"Résultats : {len(df_search)} candidats")
    
    if len(df_search) > 0:
        df_results = df_search.nlargest(20, 'composite_score')[
            ['candidate_id', 'current_position', 'experience_years', 'domain_expertise', 
             'composite_score', 'expected_salary', 'availability_weeks']
        ].copy()
        
        df_results['Score'] = df_results['composite_score'].apply(lambda x: f"{x:.1f}")
        df_results['Salaire'] = df_results['expected_salary'].apply(lambda x: f"{x:,} €")
        df_results['Exp.'] = df_results['experience_years'].apply(lambda x: f"{x:.1f} ans")
        df_results['Dispo'] = df_results['availability_weeks'].apply(lambda x: "Immédiate" if x == 0 else f"{x} sem.")
        
        st.dataframe(
            df_results[['candidate_id', 'current_position', 'Exp.', 'domain_expertise', 'Score', 'Salaire', 'Dispo']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Aucun candidat ne correspond aux critères.")

st.markdown("---")

# Fiche candidat
st.header("Fiche Candidat")

selected_candidate = st.selectbox("Sélectionner un candidat", df['candidate_id'].tolist())

if selected_candidate:
    candidate = df[df['candidate_id'] == selected_candidate].iloc[0]
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        score = candidate['composite_score']
        color = '#2ECC71' if score >= 70 else '#F39C12' if score >= 50 else '#E74C3C'
        
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0;">Score: {score:.1f}/100</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Poste:** {candidate['current_position']}")
        st.markdown(f"**Expérience:** {candidate['experience_years']:.1f} ans")
        st.markdown(f"**Domaine:** {candidate['domain_expertise']}")
        st.markdown(f"**Diplôme:** {candidate['degree']}")
    
    with col2:
        st.markdown(f"**Prétention:** {candidate['expected_salary']:,} €")
        dispo = candidate['availability_weeks']
        st.markdown(f"**Disponibilité:** {'Immédiate' if dispo == 0 else f'{dispo} semaines'}")
        st.markdown(f"**Entretien:** {candidate['interview_score']}/100")
        st.markdown(f"**Test technique:** {candidate['technical_test_score']}/100")
    
    with col3:
        # Radar des compétences
        tech_display = ['Python', 'SQL', 'Statistics', 'Excel', 'Machine Learning']
        values = [candidate[f'skill_{s}'] for s in tech_display]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=tech_display + [tech_display[0]],
            fill='toself',
            fillcolor='rgba(155, 89, 182, 0.3)',
            line=dict(color='#9B59B6', width=2)
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Compétences Techniques",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
