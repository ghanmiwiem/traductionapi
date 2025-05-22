from flask import Flask, request, jsonify, url_for
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from langdetect import detect, DetectorFactory
import re

DetectorFactory.seed = 0
app = Flask(__name__)

# تحميل النموذج السياقي
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# تحميل قاعدة البيانات
df = pd.read_csv("phrases_vedio_langue.csv")

# دالة تنظيف وتطبيع النص
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # إزالة علامات الترقيم
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة الفراغات الزائدة
    return text

# تطبيع كل الجمل في قاعدة البيانات مسبقًا
df['normalized_phrase'] = df['phrase'].apply(normalize)

# كلمات مفتاحية بسيطة لتقوية كشف اللغة
french_keywords = ['bonjour', 'merci', 'comment', 's’il vous plaît', 'au revoir', 'demain']
english_keywords = ['hello', 'thank', 'please', 'bye', 'tomorrow']

# دالة كشف اللغة
def detect_language(text):
    text = normalize(text)
    if len(text.split()) == 1 or len(text) < 5:
        if text in french_keywords:
            return 'fr'
        elif text in english_keywords:
            return 'en'
    try:
        lang = detect(text)
        return 'fr' if lang.startswith('fr') else 'en'
    except:
        return 'en'

# دالة إيجاد الفيديو الأنسب
def get_video_for_input(text, df):
    normalized_text = normalize(text)
    language = detect_language(normalized_text)
    df_lang = df[df['langue'] == language]
    words = normalized_text.split()

    # تطابق كامل
    matches = df_lang[df_lang['normalized_phrase'] == normalized_text]
    if not matches.empty:
        return [matches.iloc[0]['vedio']]

    # تطابق جزئي كلمة بكلمة
    video_paths = []
    for word in words:
        match = df_lang[df_lang['normalized_phrase'] == word]
        if not match.empty:
            video_paths.append(match.iloc[0]['vedio'])

    if video_paths:
        return video_paths

    # تطابق سياقي (ذكاء اصطناعي)
    input_embedding = semantic_model.encode(normalized_text, convert_to_tensor=True)
    phrases = df_lang['normalized_phrase'].tolist()
    embeddings = semantic_model.encode(phrases, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, embeddings)[0]
    best_match_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[best_match_idx].item()

    if best_score >= 0.5:
        return [df_lang.iloc[best_match_idx]['vedio']]
    else:
        return []

# مسار API
@app.route('/get-video', methods=['POST'])
def get_video():
    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    results = get_video_for_input(user_text, df)

    # تحويل المسارات إلى روابط كاملة من مجلد static
    full_video_urls = [
        url_for('static', filename=f"{path}.mp4", _external=True)
        for path in results
    ]

    return jsonify({"videos": full_video_urls})

# تشغيل السيرفر
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
 