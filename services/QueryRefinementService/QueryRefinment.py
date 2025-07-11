
import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk import pos_tag, word_tokenize
import json

# كلمات يجب تجاهلها
EXCLUDED_LEMMAS = {
    "do", "does", "did", "make", "thing", "something", "someone", "entity", "object",
    "give", "have", "get", "be", "take", "person", "individual", "go",
    "i", "we", "you", "he", "she", "it", "they", "how", "to", "me", "can", "tell",
    "why", "what", "where", "which", "whom", "whose", "when", "will", "would", "should",
    "could", "may", "might", "shall", "must", "is", "am", "are", "was", "were", "been", "being"
}

# أصناف الكلمات التي يُسمح بتوسيعها
ALLOWED_POS = {'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

class QueryRefinement:

    def refine_query(self, query):
        tokens = word_tokenize(query)
        tagged_tokens = pos_tag(tokens)
        refined_tokens = set()

        for token, tag in tagged_tokens:
            token_lower = token.lower()

            # استبعاد الكلمات غير المفيدة
            if token_lower in EXCLUDED_LEMMAS or token_lower.isnumeric():
                continue

            # نضيف الكلمة الأصلية
            refined_tokens.add(token_lower)

            # نوسع فقط الكلمات ذات دلالة
            if tag in ALLOWED_POS:
                synset = lesk(tokens, token)  # توسع سياقي
                if synset:
                    for lemma in synset.lemmas():
                        lemma_name = lemma.name().replace('_', ' ').lower()
                        if (lemma_name != token_lower and
                                lemma_name not in EXCLUDED_LEMMAS and
                                not lemma_name.isnumeric()):
                            refined_tokens.add(lemma_name)

        return ' '.join(sorted(refined_tokens))

    def refine_queries_file(self, input_txt_file, output_jsonl_file):
        refined_queries = []

        try:
            with open(input_txt_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        try:
                            query_id, query_text = line.strip().split('\t', 1)
                            refined_query = self.refine_query(query_text)
                            refined_queries.append({
                                "qid": query_id,
                                "query": refined_query
                            })
                            print(f"[{query_id}] ✅ تم تحسين: {query_text} ➜ {refined_query}")
                        except ValueError as e:
                            print(f"⚠️ خطأ في السطر: {line.strip()}")
                            print(f"   الرسالة: {e}")
        except FileNotFoundError:
            print(f"❌ الملف غير موجود: {input_txt_file}")
            return

        # كتابة النتائج إلى ملف JSONL
        with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
            for item in refined_queries:
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write('\n')


# تشغيل مباشر إذا كان الملف الرئيسي
if __name__ == "__main__":
    refiner = QueryRefinement()
    refiner.refine_queries_file(
        r"D:\IrProject\datasets\dataset_quore\queries_quore.txt",
        r"D:\IrProject\datasets\dataset_quore\queries_quore_refinedFinal26.jsonl"
    )
