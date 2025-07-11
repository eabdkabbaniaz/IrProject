# import re

# # تحميل الاستعلامات من الملف
# def load_queries(file_path):
#     queries = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 try:
#                     _, query = line.strip().split('\t', 1)
#                     queries.append(query.strip().lower())
#                 except ValueError:
#                     continue
#     return queries

# # دالة الإكمال التلقائي
# def autocomplete(queries, user_input, max_suggestions=10):
#     user_input = user_input.lower().strip()
#     pattern = re.compile(r'^' + re.escape(user_input))  # طابق من بداية الجملة
#     suggestions = [q for q in queries if pattern.match(q)]
#     return suggestions[:max_suggestions]

# # مثال على الاستخدام
# if __name__ == "__main__":
#     file_path = r"C:\Users\Raghad\Desktop\IR\antique_train_queries.txt" # غيري المسار حسب موقع الملف عندك
#     all_queries = load_queries(file_path)

#     while True:
#         user_input = input("\n🔍 أدخل بداية استعلام: ").strip()
#         if not user_input:
#             print("⛔ انتهى البرنامج.")
#             break
#         results = autocomplete(all_queries, user_input)
#         print("\n✅ اقتراحات الإكمال:")
#         for i, suggestion in enumerate(results, 1):
#             print(f"{i}. {suggestion}")
import re

# تحميل الاستعلامات مرة واحدة عند تشغيل الملف
def load_queries(file_path):
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    _, query = line.strip().split('\t', 1)
                    queries.append(query.strip().lower())
                except ValueError:
                    continue
    return queries

# المسار إلى ملف الاستعلامات
QUERY_FILE_PATH = r"D:\IrProject\datasets\dataset_quore\queries_quore.txt"
QUERIES = load_queries(QUERY_FILE_PATH)

# تابع الإكمال التلقائي - هذا ما سيتم ربطه بالـ API
def autocomplete(user_input, max_suggestions=10):
    user_input = user_input.lower().strip()
    pattern = re.compile(r'^' + re.escape(user_input))
    suggestions = [q for q in QUERIES if pattern.match(q)]
    return suggestions[:max_suggestions]

# # اختبار قبل التحويل إلى API
# if __name__ == "__main__":
#     test_query = "why do we"
#     results = autocomplete(test_query)

#     print("\n✅ اقتراحات الإكمال:")
#     if results:
#         for i, suggestion in enumerate(results, 1):
#             print(f"{i}. {suggestion}")
#     else:
#         print("❌ لا توجد اقتراحات.")
