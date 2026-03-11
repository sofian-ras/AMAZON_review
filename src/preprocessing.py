import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ce script fait la partie preprocessing (phase 1 + phase 2)
# entrée: fichier amazon au format .ft.txt
# sortie: csv nettoyé + json de stats

RANDOM_STATE = 42


def nettoyer_texte(
    texte: str,
    retirer_accents: bool,
    strategie_nombres: str,
    appliquer_stopwords: bool,
    appliquer_lemmatisation: bool,
) -> tuple[str, str, int]:
    # 1) nettoyage html + urls + mentions
    texte = html.unescape(texte)
    texte = re.sub(r"<[^>]+>", " ", texte)
    texte = re.sub(r"https?://\S+|www\.\S+", " ", texte)
    texte = re.sub(r"\b\S+@\S+\.\S+\b", " ", texte)
    texte = re.sub(r"@\w+", " ", texte)

    # 2) normalisation de base
    texte = texte.lower()

    # expansions utiles pour le sentiment
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "i'm": "i am",
        "i've": "i have",
        "i'll": "i will",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "we're": "we are",
        "you're": "you are",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    for forme, expansion in contractions.items():
        texte = re.sub(rf"\b{re.escape(forme)}\b", expansion, texte)

    if retirer_accents:
        normalise = unicodedata.normalize("nfd", texte)
        texte = "".join(c for c in normalise if unicodedata.category(c) != "mn")

    # 3) stratégie sur les nombres
    if strategie_nombres == "remplacer":
        texte = re.sub(r"\b\d+(?:[\.,]\d+)?\b", " <num> ", texte)
    elif strategie_nombres == "supprimer":
        texte = re.sub(r"\b\d+(?:[\.,]\d+)?\b", " ", texte)

    texte = re.sub(r"\s+", " ", texte).strip()

    # 4) tokenisation nltk
    tokens = word_tokenize(texte)

    tokens = [t for t in tokens if t.strip()]

    # 5) stopwords: on garde les mots négatifs
    if appliquer_stopwords:
        mots_negatifs_a_garder = {"not", "no", "nor", "don't", "dont", "cannot", "can"}
        stopwords_anglais = set(stopwords.words("english"))
        stopwords_filtrees = stopwords_anglais - mots_negatifs_a_garder
        tokens = [t for t in tokens if t not in stopwords_filtrees]

    # 6) stemming pour comparaison
    stemmer = PorterStemmer()
    tokens_stemmes = [stemmer.stem(t) for t in tokens]

    # 7) lemmatisation si activée
    if appliquer_lemmatisation:
        lemmatiseur = WordNetLemmatizer()
        tokens = [lemmatiseur.lemmatize(t) for t in tokens]

    texte_nettoye = " ".join(tokens)
    texte_stemme = " ".join(tokens_stemmes)

    return texte_nettoye, texte_stemme, len(tokens)


def main() -> None:
    print("\n" + "="*80)
    print("PHASE 1 : PREPROCESSING AMAZON REVIEWS")
    print("="*80)
    
    # 1) configuration et dépendances
    print("\n[1/5] téléchargement des dépendances nltk...")
    # dépendances nltk obligatoires
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    print("✓ dépendances ok")

    parseur = argparse.ArgumentParser(description="phase 1 preprocessing amazon reviews")
    parseur.add_argument("--chemin_entree", type=str, default="data/train.ft.txt", help="fichier .ft.txt (default: data/train.ft.txt)")
    parseur.add_argument("--chemin_sortie_csv", type=str, default="data/prepared/p1_preprocessing.csv")
    parseur.add_argument("--chemin_sortie_stats", type=str, default="data/prepared/p1_preprocessing_stats.json")
    parseur.add_argument("--taille_subset", type=int, default=10000)
    parseur.add_argument("--graine", type=int, default=42)
    parseur.add_argument("--retirer_accents", action="store_true")
    parseur.add_argument("--strategie_nombres", type=str, choices=["remplacer", "supprimer", "garder"], default="remplacer")
    parseur.add_argument("--sans_stopwords", action="store_true")
    parseur.add_argument("--sans_lemmatisation", action="store_true")
    arguments = parseur.parse_args()

    if arguments.graine is None:
        arguments.graine = RANDOM_STATE

    print(f"\n[2/5] lecture du fichier: {arguments.chemin_entree}...")
    # 2) chargement du dataset amazon
    # lecture du fichier amazon (.ft.txt)
    lignes_valides = []
    with open(arguments.chemin_entree, "r", encoding="utf-8", errors="ignore") as fichier:
        for numero_ligne, ligne in enumerate(fichier, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue

            # format attendu: __label__x espace texte
            if " " not in ligne or not ligne.startswith("__label__"):
                continue

            etiquette_brute, texte = ligne.split(" ", 1)
            etiquette = etiquette_brute.replace("__label__", "")

            if etiquette not in {"1", "2"}:
                continue

            sentiment = "negatif" if etiquette == "1" else "positif"
            lignes_valides.append(
                {
                    "id_ligne": numero_ligne,
                    "label": int(etiquette),
                    "sentiment": sentiment,
                    "avis_brut": texte.strip(),
                }
            )

    donnees = pd.DataFrame(lignes_valides)
    if donnees.empty:
        raise ValueError("aucune ligne exploitable dans le fichier d'entree")
    
    print(f"✓ {len(donnees):,} avis chargés")
    print(f"  - {(donnees['label']==1).sum():,} négatifs")
    print(f"  - {(donnees['label']==2).sum():,} positifs")

    print(f"\n[3/5] nettoyage brut...")
    # 3) subset équilibré + nettoyage brut
    # subset équilibré (50/50) pour temps de calcul raisonnable
    if 0 < arguments.taille_subset < len(donnees):
        taille_par_classe = arguments.taille_subset // 2
        morceaux = []
        for etiquette, groupe in donnees.groupby("label"):
            taille_reelle = min(taille_par_classe, len(groupe))
            morceaux.append(groupe.sample(n=taille_reelle, random_state=arguments.graine))
        donnees = pd.concat(morceaux, ignore_index=True)

    donnees = donnees.sample(frac=1, random_state=arguments.graine).reset_index(drop=True)

    # nettoyage brut
    nb_apres_parse = len(donnees)
    donnees = donnees.drop_duplicates(subset=["avis_brut"]).reset_index(drop=True)
    nb_apres_doublons = len(donnees)
    print(f"  • suppression doublons: {nb_apres_parse - nb_apres_doublons:,} lignes retirées")

    donnees["longueur_brut"] = donnees["avis_brut"].str.len()
    nb_avant_longueur = len(donnees)
    donnees = donnees[donnees["longueur_brut"] >= 5].copy()
    print(f"  • suppression avis trop courts (< 5 chars): {nb_avant_longueur - len(donnees):,} lignes retirées")

    # filtre bruit: supprimer avis composés presque uniquement de caractères spéciaux
    def calcul_ratio_speciaux(texte: str) -> float:
        if not texte:
            return 1.0
        total = len(texte)
        speciaux = sum(1 for c in texte if not c.isalnum() and not c.isspace())
        return speciaux / total

    donnees["ratio_speciaux"] = donnees["avis_brut"].map(calcul_ratio_speciaux)
    nb_avant_speciaux = len(donnees)
    donnees = donnees[donnees["ratio_speciaux"] <= 0.95].copy()
    print(f"  • suppression avis trop bruités (> 95% spéciaux): {nb_avant_speciaux - len(donnees):,} lignes retirées")
    print(f"✓ {len(donnees):,} avis après nettoyage brut")


    print(f"\n[4/5] nettoyage linguistique (tokenisation, lemmatisation, stopwords)...")
    # 4) nettoyage linguistique
    # nettoyage linguistique
    resultats = donnees["avis_brut"].map(
        lambda t: nettoyer_texte(
            texte=t,
            retirer_accents=arguments.retirer_accents,
            strategie_nombres=arguments.strategie_nombres,
            appliquer_stopwords=not arguments.sans_stopwords,
            appliquer_lemmatisation=not arguments.sans_lemmatisation,
        )
    )

    donnees["avis_nettoye"] = resultats.map(lambda x: x[0])
    donnees["avis_stemme"] = resultats.map(lambda x: x[1])
    donnees["nb_tokens"] = resultats.map(lambda x: x[2])
    donnees["longueur_nettoye"] = donnees["avis_nettoye"].str.len()

    # retire les lignes devenues vides après nettoyage
    nb_avant_final = len(donnees)
    donnees = donnees[donnees["avis_nettoye"].str.len() > 0].copy()
    print(f"  • suppression avis devenues vides après nettoyage: {nb_avant_final - len(donnees):,}")
    print(f"✓ nettoyage terminé")
    print(f"  • tokens moyen par avis: {donnees['nb_tokens'].mean():.1f}")


    print(f"\n[5/5] export des résultats...")
    # 5) export des résultats
    # export prêt pour les autres membres du groupe
    chemin_csv = Path(arguments.chemin_sortie_csv)
    chemin_stats = Path(arguments.chemin_sortie_stats)
    chemin_csv.parent.mkdir(parents=True, exist_ok=True)
    chemin_stats.parent.mkdir(parents=True, exist_ok=True)

    donnees.to_csv(chemin_csv, index=False, encoding="utf-8")

    stats = {
        "nb_lignes_apres_parse": int(nb_apres_parse),
        "nb_lignes_apres_doublons": int(nb_apres_doublons),
        "nb_lignes_final": int(len(donnees)),
        "distribution_labels": donnees["label"].value_counts().to_dict(),
        "longueur_brut_moyenne": float(donnees["longueur_brut"].mean()),
        "longueur_brut_min": int(donnees["longueur_brut"].min()),
        "longueur_brut_max": int(donnees["longueur_brut"].max()),
        "longueur_nettoye_moyenne": float(donnees["longueur_nettoye"].mean()),
    }

    with open(chemin_stats, "w", encoding="utf-8") as fichier_stats:
        json.dump(stats, fichier_stats, ensure_ascii=False, indent=2)

    print(f"\n✓ fichier csv généré: {chemin_csv}")
    print(f"✓ fichier stats généré: {chemin_stats}")
    print(f"\nRÉSUMÉ:")
    print(f"  - {len(donnees):,} avis nettoyés et prêts")
    print(f"  - {donnees[donnees['label']==1].shape[0]:,} négatifs")
    print(f"  - {donnees[donnees['label']==2].shape[0]:,} positifs")
    print(f"  - longueur moyenne (nettoyé): {donnees['longueur_nettoye'].mean():.0f} caractères")
    print(f"\n" + "="*50)
    print("✓ preprocessing terminé! prêt pour les features et modèles")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
