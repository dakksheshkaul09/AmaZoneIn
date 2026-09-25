import pandas as pd
import unicodedata


def normalize_name(x):
    if pd.isna(x):
        return ""

    x = str(x).casefold()

    # Normalize Unicode characters
    x = unicodedata.normalize("NFKD", x)

    # Remove accents:
    # Récord -> Record
    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )

    # Replace punctuation with spaces
    x = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in x
    )

    # Remove extra whitespace
    x = " ".join(x.split())

    return x