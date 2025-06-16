import json

# مسار الفهرس المعكوس
inverted_index_path = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\inverted_index.json"

# تحميل الفهرس
try:
    print(f"📁 Trying to open: {inverted_index_path}")
    with open(inverted_index_path, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
except Exception as e:import json

inverted_index_path = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\inverted_index.json"

try:
    print(f"📁 Trying to open: {inverted_index_path}")
    with open(inverted_index_path, 'r', encoding='utf-8', errors='ignore') as file:
        data = file.read()
        inverted_index = json.loads(data)
except Exception as e:
    print(f"❌ خطأ في فتح أو تحميل الفهرس: {e}")
    exit()

# عرض أول 30 مصطلح
print("📚 أول 30 مصطلح في الفهرس المعكوس:\n")
for i, (term, doc_ids) in enumerate(inverted_index.items()):
    print(f"{i+1}. '{term}': {doc_ids[:5]}{'...' if len(doc_ids) > 5 else ''}")
    if i >= 29:
        break

    print(f"❌ خطأ في فتح أو تحميل الفهرس: {e}")
    exit()

# عرض أول 30 مصطلح
print("📚 أول 30 مصطلح في الفهرس المعكوس:\n")
for i, (term, doc_ids) in enumerate(inverted_index.items()):
    print(f"{i+1}. '{term}': {doc_ids[:5]}{'...' if len(doc_ids) > 5 else ''}")
    if i >= 29:
        break
