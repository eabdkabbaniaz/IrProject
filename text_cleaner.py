# text_cleaner.py
import pandas as pd
import logging
from TextProcessing.TextProcessing import TextProcessor, process_text

logging.basicConfig(level=logging.INFO)

def clean_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path, sep='\t', on_bad_lines='warn', header=None, names=["id", "text"])
    processor = TextProcessor()

    processed = []
    for _, row in df.iterrows():
        if pd.isna(row["text"]): continue
        cleaned = process_text(row["text"], processor)
        processed.append([row["id"], cleaned])

    df_cleaned = pd.DataFrame(processed, columns=["id", "text"])
    df_cleaned.to_csv(output_path, sep='\t', index=False)
    logging.info(f"Cleaned data saved to {output_path}")
    return output_path
