# app.py

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import difflib
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

app = Flask(__name__)

# تحميل النموذج السياقي
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# تحميل قاعدة البيانات
df = pd.read_csv("phrases_vedio_langue.csv")  # غيّر اسم الملف حسب اسم قاعدة بياناتك

# كلمات مفتاحية للغة
french_keywords = ['bonjour', 'merci', 'comment', 's’il vous plaît', 'au revoir']
english_keywords = ['hello', 'thank', 'please', 'bye']

def detect_language(text):
    text = text.lower().strip()
    if len(text.split()) == 1 or len(text) < 5:
        if text in french_keywords:
            return 'fr'
        elif text in english_keywords:
            return 'en'
    try:
        lang = detect(text)
        if lang.startswith('fr'):
            return 'fr'
        elif lang.startswith('en'):
            return 'en'
        else:
            return 'en'
    except:
        return 'en'

def get_video_for_input(text, df):
    text = text.lower().strip()
    language = detect_language(text)
    df_lang = df[df['langue'] == language]
    words = text.split()

    # تطابق كامل
    matches = df_lang[df_lang['phrase'].str.lower() == text]
    if not matches.empty:
        return [matches.iloc[0]['vedio']]

    # تطابق جزئي
    video_paths = []
    for word in words:
        match = df_lang[df_lang['phrase'].str.lower() == word]
        if not match.empty:
            video_paths.append(match.iloc[0]['vedio'])

    if video_paths:
        return video_paths

    # تطابق سياقي
    input_embedding = semantic_model.encode(text, convert_to_tensor=True)
    phrases = df_lang['phrase'].tolist()
    embeddings = semantic_model.encode(phrases, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, embeddings)[0]
    best_match_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[best_match_idx].item()

    if best_score >= 0.5:
        return [df_lang.iloc[best_match_idx]['vedio']]
    else:
        return []

@app.route('/get-video', methods=['POST'])
def get_video():
    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    results = get_video_for_input(user_text, df)
    return jsonify({"videos": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

