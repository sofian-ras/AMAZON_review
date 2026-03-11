# sentiment analysis amazon reviews

## structure du projet

```
amazon_rvw/
├── src/
│   └── preprocessing.py          # phase 1 + 2: nettoyage des données
├── scripts/
│   ├── visualiser_overview.py    # affiche stats globales
│   └── visualiser_exemples.py    # affiche exemples textes
├── notebooks/
│   └── (phase3_features.ipynb, phase4_modeles.ipynb, etc.)
├── data/
│   ├── raw/
│   │   └── train.ft.txt          # données brutes (in .gitignore)
│   └── prepared/
│       ├── p1_preprocessing.csv   # sortie nettoyée
│       └── p1_preprocessing_stats.json
├── README.md
└── TP-GROUPE-amazon-reviews-nlp.md
```

## démarrage rapide

### phase 1: preprocessing (ta partie)

```bash
# lancer le nettoyage
python -m src.preprocessing

# ou avec options
python -m src.preprocessing --chemin_entree "data/raw/train.ft.txt" --taille_subset 10000
```

### visualiser les résultats

```bash
python scripts/visualiser_overview.py
python scripts/visualiser_exemples.py
```

## options du preprocessing

```
--chemin_entree          chemin du fichier .ft.txt (default: train.ft.txt)
--chemin_sortie_csv      où sauvegarder le csv nettoyé
--taille_subset          nombre d'avis à traiter (default: 10000)
--graine                 seed reproductible (default: 42)
--retirer_accents        enlever les accents (optionnel)
--strategie_nombres      remplacer/supprimer/garder (default: remplacer)
--sans_stopwords         ne pas enlever les stopwords
--sans_lemmatisation     ne pas lemmatiser
```

## sorties

- `data/prepared/p1_preprocessing.csv`: dataset nettoyé avec colonnes:
  - `label`, `sentiment`: 1=négatif, 2=positif
  - `avis_brut`: texte original
  - `avis_nettoye`: texte après lemmatisation
  - `avis_stemme`: texte après stemming
  - `nb_tokens`, `longueur_brut`, `longueur_nettoye`: métadonnées

- `data/prepared/p1_preprocessing_stats.json`: stats globales du preprocessing

## pour les autres membres du groupe

### phase 2: features engineering
charger le csv et créer des vectorisations (TF-IDF, Word2Vec, etc.)

### phase 3: modélisation
tester différents modèles (LR, SVM, RF, etc.)

### phase 4: validation et analyse
évaluer les performances et analyser les erreurs

## dépendances

```
pandas
nltk
numpy
```

## git workflow

```bash
# après changements
git add src/ scripts/ notebooks/ data/prepared/
git commit -m "phase 1: preprocessing nettoyage données"
git push origin main
```

**note**: `train.ft.txt` est dans `.gitignore` (trop volumineux)
