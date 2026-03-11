import pandas as pd

df = pd.read_csv("data/prepared/p1_preprocessing.csv")

print("=" * 100)
print("TEXTES BRUTS vs NETTOYÉS - PHASE 1")
print("=" * 100)

for i in range(3):
    print(f"\n--- EXEMPLE {i+1} ---")
    print(f"Label: {df.iloc[i]['sentiment'].upper()}")
    print(f"\nTexte BRUT ({df.iloc[i]['longueur_brut']} caractères):")
    print(f"  {df.iloc[i]['avis_brut'][:150]}...")
    print(f"\nTexte NETTOYÉ ({df.iloc[i]['nb_tokens']} tokens):")
    print(f"  {df.iloc[i]['avis_nettoye'][:150]}...")
    print(f"\nTexte STEMMÉ:")
    print(f"  {df.iloc[i]['avis_stemme'][:150]}...")
