# Projet PPE2 – Traitement de flux RSS et Topic Modeling

**Auteurs :** Sasha LATIMIER, Jourdan WILSON, Xingyu CHEN, Yongcan LIU  
**Branche principale :** `main`  
**Documentation générée (MkDocs) :** branche `doc` ([GitLab Pages](https://projet-38fbac.gitlab.io))  

---

## Présentation

Ce projet implémente un pipeline de **traitement de flux RSS** :
1. **Extraction** récursive d’articles XML/JSON/Pickle
2. **Analyse linguistique** (lemmatisation, morphosyntaxe via spaCy)
3. **Filtrage** par source, catégorie, date, POS…
4. **Topic Modeling** (BERTopic)
5. **Visualisations** 

Il s’articule en deux modules :
- `src/` : scripts Python et utilitaires
- `src/notebooks/` : notebooks Jupyter d’expérimentation (BERTopic, Gensim, LDA…)

---


## Arborescence du projet

```
.
├── README.md              ← Ce fichier de présentation
├── requirements.txt       ← Liste des dépendances Python
├── src/                   ← Code source principal
│   ├── analyzers.py
│   ├── datastructures.py
│   ├── pipeline.py
│   ├── rss_parcourir.py
│   ├── rss_reader.py
│   ├── stopwords-fr.txt
│
└── src/notebooks/         ← Jupyter notebooks d’exploration
    ├── bertopic_start.ipynb
    ├── gensim.ipynb
    └── run_lda.py

```

### Branch `doc`
- Contient la documentation générée avec MkDocs dans `docs/`.
- Fichier de configuration `.gitlab-ci.yml` pour déployer la doc sur GitLab Pages.

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://gitlab.com/plurital-ppe2-2025/groupe09/Projet.git
   cd Projet
   ```

2. Créer un environnement virtuel et installer les dépendances :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. (Optionnel) Installer MkDocs pour la documentation :
   ```bash
   pip install mkdocs
   ```

## Usage

### Extraction et filtrage RSS
```bash
python src/rss_parcourir.py --start 2025-01-01 --categories Science,Politique --source LeMonde
python src/unified_split_or_global.py --mode split --split_type source
```

### Topic Modeling
```bash
python pipeline.py path/to/rss_folder --reader etree --walker pathlib --start 01/05/25 --categories finance crypto --source "BFM CRYPTO" --analyzer spacy --n_topics 15 --n_components 10 --min_dist 0.1 --min_cluster_size 8 --min_samples 8 --save_embeddings --pos NOUN --format json --output ./complete_results
```

### Visualisations
Les résultats sont générés dans le dossier `docs/visualizations/…` et accessibles via le site MkDocs (branche `doc`).

## Documentation

Le site de documentation (MkDocs) est déployé sur la branche `doc` via GitLab Pages. Pour le consulter localement :
```bash
mkdocs serve
```
→ Ouvrir `http://127.0.0.1:8000/`

## Contribuer

- Créez une branche pour votre fonctionnalité : `git checkout -b feature/mon-idee`
- Validez vos modifications : `git commit -am "Ajout de la fonctionnalité X"`
- Poussez et ouvrez une merge request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
