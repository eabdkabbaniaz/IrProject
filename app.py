from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

FASTAPI_SUGGEST_URL = "http://127.0.0.1:8000/suggest_query"
FASTAPI_AUTOCOMPLETE_URL = "http://127.0.0.1:8000/autoComplete_query"
FASTAPI_CORRECT_URL = "http://127.0.0.1:8000/correct_query"

FASTAPI_TFIDF_URL = "http://127.0.0.1:8000/applaySearchTfidf"
FASTAPI_BERT_URL = "http://127.0.0.1:8000/preview-BertModelSearch"
FASTAPI_SEQUENTIAL_URL = "http://127.0.0.1:8000/preview-HybirdSequential"

FASTAPI_BM25_URL = "http://127.0.0.1:8000/preview-BM25Search"
FASTAPI_PARALLEL_URL = "http://127.0.0.1:8000/preview-HybirdParallel"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/basic")
def basic():
     return render_template(
        "basic.html",
        tfidf_url=FASTAPI_TFIDF_URL,
        bert_url=FASTAPI_BERT_URL,
        sequential_url=FASTAPI_SEQUENTIAL_URL
    )

@app.route("/additional2")
def additional2():
     return render_template(
        "additional2.html",
        bm25_url=FASTAPI_BM25_URL,
        parallel_url=FASTAPI_PARALLEL_URL
    )

@app.route("/additional", methods=["GET", "POST"])
def additional():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        dataset = request.form.get("dataset", "")

        cleaned_query = query.lower()

        try:
            response = requests.post(FASTAPI_SUGGEST_URL, json={"query": cleaned_query})
            response.raise_for_status()  # لو فيه خطأ status_code راح يعمل استثناء
            response_data = response.json()
            tfidf_results = response_data.get("message", "لم يتم الحصول على نتائج")
        except Exception as e:
            tfidf_results = f"حدث خطأ أثناء الاتصال بـ FastAPI: {e}"

        return render_template(
            "additional.html",
            original_query=query,
            dataset=dataset,
            cleaned_query=cleaned_query,
            tfidf_results=tfidf_results
        )

    return render_template("additional.html")

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"message": ["الاستعلام فارغ."]}), 400

        fastapi_response = requests.post(FASTAPI_AUTOCOMPLETE_URL, json={"query": query})
        fastapi_response.raise_for_status()
        return jsonify(fastapi_response.json())

    except requests.exceptions.ConnectionError:
        return jsonify({"message": ["تعذر الاتصال بـ FastAPI. تأكد أنه يعمل."]}), 503

    except Exception as e:
        return jsonify({"message": [f"خطأ أثناء الإكمال التلقائي: {e}"]}), 500
    
@app.route("/correct_query", methods=["POST"])
def crrectQuery():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"message": ["الاستعلام فارغ."]}), 400

        fastapi_response = requests.post(FASTAPI_CORRECT_URL, json={"query": query})
        fastapi_response.raise_for_status()
        return jsonify(fastapi_response.json())

    except requests.exceptions.ConnectionError:
        return jsonify({"message": ["تعذر الاتصال بـ FastAPI. تأكد أنه يعمل."]}), 503

    except Exception as e:
        return jsonify({"message": [f"خطأ أثناء الإكمال التلقائي: {e}"]}), 500

if __name__ == "__main__":
    app.run(debug=True)
