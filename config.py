# config.py  ← à la racine
RANDOM_STATE = 42
SAMPLE_SIZE  = 5000
TEST_SIZE    = 0.2

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.8,
}