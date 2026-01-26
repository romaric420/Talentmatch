# TalentMatch - Scoring de Candidatures RH

Systeme intelligent de matching et scoring de candidats pour le recrutement.

## Fonctionnalites

- **Tableau de bord RH** : Vue d'ensemble du vivier de candidats
- **Recherche avancee** : Filtrage multi-criteres avec scoring en temps reel
- **Matching KNN** : Identification de profils similaires
- **Segmentation** : Clustering et analyse des talents

## Algorithmes Implementes

### Scoring Multi-criteres
Systeme de scoring parametrable avec ponderations ajustables:
- Competences techniques (Python, SQL, ML, etc.)
- Experience professionnelle
- Soft skills (communication, travail equipe)
- Evaluations (entretiens, tests techniques)

### Matching KNN
Utilisation de K-Nearest Neighbors pour:
- Trouver des profils similaires a un candidat cible
- Benchmarking de competences
- Identification de candidats de substitution

### Segmentation PCA
- Reduction de dimensionnalite pour visualisation
- Clustering des profils de candidats
- Identification des segments de talents

## Metriques Candidat

| Categorie | Metriques |
|-----------|-----------|
| Technique | Score par competence (0-100) |
| Experience | Annees, projets, certifications |
| Evaluation | Entretien, test technique, culture fit |
| Disponibilite | Delai, mobilite, teletravail |

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure du Code

```python
# Fonctions principales
generate_candidate_data()     # Generation vivier candidats
build_knn_model()            # Modele de matching
calculate_composite_score()  # Scoring parametrable
plot_candidate_radar()       # Visualisation profil
plot_pca_candidates()        # Clustering
```

## Cas d'Usage

1. **Recruteur** : Identifier les meilleurs candidats selon criteres
2. **RH** : Analyser le vivier et les tendances
3. **Manager** : Comparer des candidats finalistes
4. **Talent Acquisition** : Benchmarking salaires/competences

## Technologies

- Python 3.9+
- Streamlit 1.31
- Scikit-learn 1.4
- Plotly 5.18
- Pandas 2.1
# Talentmatch
