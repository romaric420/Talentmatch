"""
Analyse et Segmentation - TalentMatch
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Analyse - TalentMatch", layout="wide")


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
    
    # Segmentation
    df['segment'] = pd.cut(
        df['composite_score'],
        bins=[0, 40, 55, 70, 100],
        labels=['À développer', 'Potentiel', 'Performant', 'Excellence']
    )
    
    return df


# PAGE


st.title("Analyse et Segmentation")
st.markdown("Segmentation des talents et visualisation par PCA")

df = load_data()

st.markdown("---")

# Segmentation
st.header("Segmentation des Talents")

col1, col2 = st.columns(2)

with col1:
    # Répartition des segments
    segment_counts = df['segment'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=segment_counts.index,
        values=segment_counts.values,
        hole=0.4,
        marker_colors=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']
    )])
    fig.update_layout(title="Répartition des Segments", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Stats par segment
    segment_stats = df.groupby('segment').agg({
        'candidate_id': 'count',
        'composite_score': 'mean',
        'experience_years': 'mean',
        'expected_salary': 'mean'
    }).round(1)
    
    segment_stats.columns = ['Nombre', 'Score Moyen', 'Exp. Moy.', 'Salaire Moy.']
    segment_stats['Salaire Moy.'] = segment_stats['Salaire Moy.'].apply(lambda x: f"{x:,.0f} €")
    
    st.dataframe(segment_stats, use_container_width=True)

st.markdown("---")

# PCA Visualization
st.header("Visualisation PCA des Candidats")

# Préparation des features pour PCA
feature_cols = [f'skill_{s}' for s in TECH_SKILLS[:6]] + ['experience_years', 'interview_score', 'technical_test_score']

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'domain': df['domain_expertise'],
    'score': df['composite_score'],
    'segment': df['segment']
})

fig = px.scatter(
    df_pca,
    x='PC1',
    y='PC2',
    color='segment',
    size='score',
    hover_data=['score'],
    opacity=0.7,
    color_discrete_map={
        'À développer': '#E74C3C',
        'Potentiel': '#F39C12',
        'Performant': '#3498DB',
        'Excellence': '#2ECC71'
    }
)

total_var = (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100
fig.update_layout(
    title=f"Clustering des Candidats (PCA - {total_var:.1f}% variance)",
    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Heatmap compétences par domaine
st.header("Compétences par Domaine")

skill_cols = [f'skill_{s}' for s in TECH_SKILLS[:6]]
heatmap_data = df.groupby('domain_expertise')[skill_cols].mean()
heatmap_data.columns = [col.replace('skill_', '') for col in heatmap_data.columns]

fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Purples',
    text=heatmap_data.values.round(0),
    texttemplate='%{text}'
))
fig.update_layout(title="Niveau Moyen par Domaine", height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Score par diplôme
st.header("Score Moyen par Diplôme")

degree_scores = df.groupby('degree')['composite_score'].mean().sort_values(ascending=True)

fig = go.Figure(go.Bar(
    x=degree_scores.values,
    y=degree_scores.index,
    orientation='h',
    marker_color='#9B59B6',
    text=degree_scores.round(1),
    textposition='outside'
))
fig.update_layout(title="Score Moyen par Niveau de Diplôme", xaxis_title="Score moyen", height=400)
st.plotly_chart(fig, use_container_width=True)
