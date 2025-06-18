import csv
import logging
import pandas as pd
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from spellchecker import SpellChecker

import inflect
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import unicodedata
import contractions
from textblob import TextBlob
import spacy

# إعداد التسجيل (logging) لتتبع الأخطاء
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# تحميل نموذج spaCy
nlp = spacy.load('en_core_web_sm')

# تنزيل بيانات NLTK المطلوبة
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# تحميل كلمات الإزالة (افتراضًا أنها في ملف منفصل)
try:
    with open(r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\stop_words.txt", 'r', encoding='utf-8') as file:
        words_to_remove = file.read().splitlines()
except FileNotFoundError:
    logging.warning("ملف stop_words.txt غير موجود، سيتم استخدام قائمة فارغة")
    words_to_remove = []

class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.inflect_engine = inflect.engine()
        self.stop_words = set(stopwords.words('english')).union(set(words_to_remove))
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def spelling_correction(self, text):
        corrected_words = []
        words = self.tokenizer.tokenize(text)
        for word in words:
            corrected_word = self.spell_checker.correction(word)
            if corrected_word is not None:
                corrected_words.append(corrected_word)
        return ' '.join(corrected_words)

    def clean_text(self, text, words_to_remove):
        words = text.split()
        cleaned_words = [word for word in words if word not in words_to_remove]
        return ' '.join(cleaned_words)

    def number_to_words(self, text):
        words = self.tokenizer.tokenize(text)
        converted_words = []
        for word in words:
            # Check if the word is purely numeric (integer)
            if word.isdigit():
                try:
                    num = int(word)
                    # You had a large number check, keep it if relevant, or simplify
                    if num <= 999999999999999: # Ensure it's within inflect's reasonable range
                        converted_word = self.inflect_engine.number_to_words(num) # Pass the integer
                        converted_words.append(converted_word)
                    else:
                        converted_words.append("[Number Out of Range]")
                except inflect.NumOutOfRangeError:
                    converted_words.append("[Number Out of Range]")
                except ValueError: # Catch potential errors if int() conversion fails unexpectedly
                    converted_words.append(word) # Append original if conversion fails
            else:
                converted_words.append(word) # If not a digit, append as is
    
        return ' '.join(converted_words)

    def remove_html_tags(self, text):
        try:
            if '<' in text and '>' in text:
                return BeautifulSoup(text, "html.parser").get_text()
            return text
        except MarkupResemblesLocatorWarning:
            logging.warning("MarkupResemblesLocatorWarning: الإدخال يشبه اسم ملف أكثر من كونه ترميزًا.")
            return text

    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKD", text)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def cleaned_text(self, text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalization_example(self, text):
        return text.lower()

    def stemming_example(self, text):
        words = self.tokenizer.tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def lemmatization_example(self, text):
        words = self.tokenizer.tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    def remove_stopwords(self, text):
        words = self.tokenizer.tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
  

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_special_characters_and_emojis(self, text):
        return re.sub(r'[^A-Za-z0-9\s]+', '', text)

    def context_aware_spelling_correction(self, text):
        return str(TextBlob(text).correct())

    def replace_synonyms(self, text):
        words = self.tokenizer.tokenize(text)
        synonym_words = [self.get_synonym(word) for word in words]
        return ' '.join(synonym_words)

    def get_synonym(self, word):
        synonyms = wordnet.synsets(word)
        if synonyms:
            return synonyms[0].lemmas()[0].name()
        return word

    def handle_negations(self, text):
        words = self.tokenizer.tokenize(text)
        negated_text = []
        negate = False
        for word in words:
            if word.lower() in ['not', "n't"]:
                negate = True
            elif negate:
                negated_text.append(f"NOT_{word}")
                negate = False
            else:
                negated_text.append(word)
        return ' '.join(negated_text)

    def remove_non_english_words(self, text):
        words = self.tokenizer.tokenize(text)
        english_words = [word for word in words if wordnet.synsets(word)]
        return ' '.join(english_words)

# def process_text(text, processor):
#     if text is None or pd.isna(text):
#         return ""
#     text = str(text)
#     text = processor.cleaned_text(text)
#     text = processor.normalization_example(text)
#     words_to_remove = set(w.lower() for w in file.read().splitlines())
#     text = processor.remove_stopwords(text)
   


# def process_text(text, processor):
#     if text is None or pd.isna(text):
#         return ""

#     text = str(text)
#     text = processor.expand_contractions(text)
#     text = processor.remove_urls(text)
#     text = processor.remove_html_tags(text)
#     text = processor.normalize_unicode(text)
#     text = processor.cleaned_text(text)
#     text = processor.normalization_example(text)      # lower
#     text = processor.remove_punctuation(text)
#     text = processor.clean_text(text, words_to_remove)
#     text = processor.remove_stopwords(text)           # بعد punctuation و lowercase
#     text = processor.handle_negations(text)
#     text = processor.number_to_words(text)
#     text = processor.spelling_correction(text)
#     text = processor.stemming_example(text)           # أو lemmatization، اختر واحد فقط
#     return text
# def process_text(text, processor):
#     if text is None or pd.isna(text):
#         return ""
#     text = str(text)
#     text = processor.expand_contractions(text)
#     text = processor.remove_html_tags(text)
#     text = processor.normalize_unicode(text)
#     text = processor.remove_urls(text)
#     text = processor.cleaned_text(text)
#     text = processor.normalization_example(text)  # lowercase
#     text = processor.remove_punctuation(text)
#     text = processor.remove_stopwords(text)       # ← هنا
#     text = processor.number_to_words(text)
#     text = processor.handle_negations(text)
#     text = processor.remove_special_characters_and_emojis(text)
#     text = processor.stemming_example(text)
#     text = processor.lemmatization_example(text)
#     return text

# def process_text(text, processor):
#     if text is None or pd.isna(text):
#         return ""
#     text = str(text)
#     text = processor.expand_contractions(text)
#     text = processor.remove_html_tags(text)
#     text = processor.normalize_unicode(text)
#     text = processor.remove_urls(text)
#     text = processor.cleaned_text(text)
#     text = processor.normalization_example(text)        # lowercase
#     text = processor.remove_punctuation(text)
#     text = processor.remove_stopwords(text)             # حذف الكلمات بعد lowercase/punctuation
#     text = processor.number_to_words(text)
#     text = processor.handle_negations(text)
#     text = processor.remove_special_characters_and_emojis(text)
    
#     # استخدم فقط واحد:
#     # text = processor.stemming_example(text)
#     text = processor.lemmatization_example(text)

#     return text

def process_text(text, processor):
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    text = processor.expand_contractions(text)
    text = processor.remove_html_tags(text)
    text = processor.normalize_unicode(text)
    text = processor.remove_urls(text)
    text = processor.cleaned_text(text)
    text = processor.normalization_example(text)        # lowercase
    text = processor.remove_punctuation(text)
    text = processor.number_to_words(text)
    text = processor.handle_negations(text)
    text = processor.remove_special_characters_and_emojis(text)
    text = processor.lemmatization_example(text)         # أو stemming، واحد فقط
    # إزالة الستوب وورد هنا بعد التعديلات كلها
    text = processor.remove_stopwords(text)
    return text

def main():
    input_file = r'C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\antique_documents.csv'
    output_file = r'C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\resualt.csv'
    
    # قراءة ملف CSV باستخدام pandas
    try:
        df = pd.read_csv(input_file, sep='\t', quoting=csv.QUOTE_ALL, on_bad_lines='warn')
    except Exception as e:
        logging.error(f"خطأ أثناء قراءة ملف CSV: {e}")
        return
    
    text_processor = TextProcessor()
    processed_rows = []
    
    # معالجة كل صف
    for index, row in df.iterrows():
        try:
            doc_id = None
            text = None
            # التحقق مما إذا كان العمود الوحيد يحتوي على "doc_id,text"
            if len(row) == 1 and 'doc_id,text' in row and pd.notna(row['doc_id,text']):
                # محاولة تقسيم العمود الوحيد إلى doc_id و text
                parts = row['doc_id,text'].split(',', 1)
                if len(parts) == 2:
                    doc_id = parts[0].strip()
                    text = parts[1].strip().strip('"')
                else:
                    logging.warning(f"تخطي الصف {index} بسبب تنسيق غير صحيح: {row.to_dict()}")
                    continue
            elif len(row) >= 2 and pd.notna(row[1]):
                # إذا كانت الصفوف تحتوي على عمودين أو أكثر
                doc_id = str(row[0])
                text = str(row[1])
            else:
                logging.warning(f"تخطي الصف {index} بسبب بيانات غير كافية: {row.to_dict()}")
                continue
            
            # معالجة النص
            processed_text = process_text(text, text_processor)
            processed_rows.append([doc_id, processed_text])
            print(processed_text + '\n')
        except Exception as e:
            logging.error(f"خطأ أثناء معالجة الصف {index}: {e}")
    
    # كتابة النتائج إلى ملف CSV
    try:
        pd.DataFrame(processed_rows, columns=['ID', 'Processed_Text']).to_csv(output_file, sep='\t', index=False)
        logging.info(f"تم كتابة النتائج إلى {output_file}")
    except Exception as e:
        logging.error(f"خطأ أثناء كتابة ملف الإخراج: {e}")

if __name__ == '__main__':
    main()
