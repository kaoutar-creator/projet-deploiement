# Projet Maliki — Refonte de déploiement ML

Ce dépôt contient une version remaniée d'un projet de scoring du risque de crédit.

Structure clé:
- `data/` : emplacement attendu du fichier `Risque_data.xls` (le jeu de données original).
- `models/` : modèles sauvegardés (versionnés).
- `src/` : utilitaires, config et fonctions réutilisables.
- `deployment/` : Dockerfile multi-stage, entrypoint, .dockerignore.
- `notebooks/` : notebook Jupyter avec workflow ML revu.

Déploiement local rapide:
1. Placer `Risque_data.xls` dans le dossier `data/`.
2. Construire l'image Docker:
```
docker-compose build
docker-compose up
```
3. Accéder à l'application sur `http://localhost:8501`.

Pour entraîner localement le modèle, utiliser le notebook `notebooks/ML_pipeline.ipynb`.
