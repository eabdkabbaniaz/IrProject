
from textblob import TextBlob
from gramformer import Gramformer


gf = Gramformer(models=1, use_gpu=False)


def safe_spell_correct(query):
    blob = TextBlob(query)
    corrected = str(blob.correct())
    return corrected


def correct_with_gramformer(query):
    corrected = gf.correct(query, max_candidates=1)
    return list(corrected)[0] if corrected else query


def full_correct_query(query):
    # print("\n" + "=" * 60)
    # print(f"🔤 الاستعلام الأصلي : {query}")

    # # 1. تصحيح الإملاء
    spelling_corrected = safe_spell_correct(query)
    # print(f"✏️ بعد تصحيح الإملاء : {spelling_corrected}")

    # 2. تصحيح القواعد
    grammar_corrected = correct_with_gramformer(spelling_corrected)
    print(f"🧠 بعد تصحيح القواعد : {grammar_corrected}")
    print("=" * 60)
    return grammar_corrected


query1 = "What can I do for more energrr?"
full_correct_query(query1)

# query2 = "why dose i have loer stomache paim?"
# full_correct_query(query2)
