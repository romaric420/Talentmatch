"""
Dashboard - TalentMatch
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard - TalentMatch", layout="wide")


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
    
    # Scores compétences techniques
    for skill in TECH_SKILLS:
        base = np.random.normal(50, 25, n)
        exp_bonus = df['experience_years'] * 2
        df[f'skill_{skill}'] = (base + exp_bonus).clip(0, 100).round(0).astype(int)
    
    # Scores soft skills
    for skill in SOFT_SKILLS:
        df[f'soft_{skill}'] = np.random.normal(65, 20, n).clip(0, 100).round(0).astype(int)
    
    # Salaire
    df['expected_salary'] = (
        32000 + 
        df['experience_years'] * 3500 +
        (df['degree'] == 'Bac+5 École').astype(int) * 8000 +
        np.random.normal(0, 5000, n)
    ).clip(28000, 120000).astype(int)
    
    # Score composite
    df['composite_score'] = (
        (df['skill_Python'] + df['skill_SQL'] + df['skill_Statistics']) / 3 * 0.4 +
        df['experience_years'] * 5 * 0.25 +
        (df['soft_Communication'] + df['soft_Travail équipe']) / 2 * 0.2 +
        (df['interview_score'] + df['technical_test_score']) / 2 * 0.15
    ).round(1)
    
    return df


# PAGE


st.title("Dashboard RH")
st.markdown("Vue d'ensemble du vivier de candidats")

df = load_data()

# KPIs
st.header("Indicateurs Clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Candidats", f"{len(df):,}")

with col2:
    st.metric("Score Moyen", f"{df['composite_score'].mean():.1f}/100")

with col3:
    st.metric("Expérience Moyenne", f"{df['experience_years'].mean():.1f} ans")

with col4:
    st.metric("Salaire Médian", f"{df['expected_salary'].median():,.0f} €")

st.markdown("---")

# Top 5 candidats
st.header("Top 5 Candidats")

top5 = df.nlargest(5, 'composite_score')[
    ['candidate_id', 'current_position', 'experience_years', 'domain_expertise', 'expected_salary', 'composite_score']
].copy()

top5['Score'] = top5['composite_score'].apply(lambda x: f"{x:.1f}/100")
top5['Salaire'] = top5['expected_salary'].apply(lambda x: f"{x:,} €")
top5['Exp.'] = top5['experience_years'].apply(lambda x: f"{x:.1f} ans")

st.dataframe(
    top5[['candidate_id', 'current_position', 'Exp.', 'domain_expertise', 'Salaire', 'Score']],
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Graphiques
col1, col2 = st.columns(2)

with col1:
    # Distribution des scores
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['composite_score'], nbinsx=20, marker_color='#9B59B6'))
    fig.update_layout(title="Distribution des Scores", xaxis_title="Score", yaxis_title="Nombre", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Répartition par domaine
    domain_counts = df['domain_expertise'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=domain_counts.index, values=domain_counts.values, hole=0.4)])
    fig.update_layout(title="Répartition par Domaine", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Experience vs Salaire
st.header("Expérience vs Salaire")

import plotly.express as px
fig = px.scatter(
    df,
    x='experience_years',
    y='expected_salary',
    color='degree',
    hover_data=['candidate_id', 'current_position'],
    opacity=0.6
)
fig.update_layout(xaxis_title="Années d'expérience", yaxis_title="Salaire (€)", height=450)
st.plotly_chart(fig, use_container_width=True)
