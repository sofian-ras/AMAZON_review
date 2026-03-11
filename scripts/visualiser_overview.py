import pandas as pd

# charger le csv généré
df = pd.read_csv("data/prepared/p1_preprocessing.csv")

print("=" * 80)
print("APPERÇU DES DONNÉES NETTOYÉES - PHASE 1")
print("=" * 80)

print(f"\nShape du dataset: {df.shape}")
print(f"Colonnes: {list(df.columns)}\n")

print("Premières lignes:")
print(df.head(10))

print("\n" + "=" * 80)
print("STATISTIQUES")
print("=" * 80)
print(df.describe())

print("\nDistribution des labels:")
print(df["label"].value_counts())

print(f"\nNombre total de lignes: {len(df)}")
