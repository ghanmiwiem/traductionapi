from flask import Flask, request, jsonify, url_for
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from langdetect import detect, DetectorFactory
import os
import re
from rapidfuzz import process, fuzz

DetectorFactory.seed = 0
app = Flask(__name__)

# تحميل النموذج السياقي
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# تحميل قاعدة البيانات
df = pd.read_csv("phrases_vedio_langue.csv")

# مجلد لتخزين الفيديوهات محليًا (تأكد من إنشائه)
VIDEO_FOLDER = "./videos"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# كلمات مفتاحية للغات
french_keywords = ['bonjour', 'merci', 'comment', 's’il vous plaît', 'au revoir']
english_keywords = ['hello', 'thank', 'please', 'bye']

def normalize_text(text):
    # تحويل النص لحروف صغيرة، إزالة علامات الترقيم والمسافات الزائدة
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # إزالة علامات الترقيم
    text = re.sub(r'\s+', ' ', text)     # إزالة المسافات الزائدة
    return text

# دالة كشف اللغة مع التطبيع
def detect_language(text):
    text = normalize_text(text)
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

# دالة إيجاد الفيديو الأنسب
def get_video_for_input(text, df):
    text_norm = normalize_text(text)
    language = detect_language(text_norm)
    df_lang = df[df['langue'] == language].copy()

    # طبعاً طبع العمود phrase بنفس التطبيع
    df_lang['phrase_norm'] = df_lang['phrase'].apply(normalize_text)

    # 1. تطابق كامل بعد التطبيع
    matches = df_lang[df_lang['phrase_norm'] == text_norm]
    if not matches.empty:
        return [matches.iloc[0]['vedio']]

    # 2. تطابق جزئي لكل كلمة
    words = text_norm.split()
    video_paths = []
    for word in words:
        match = df_lang[df_lang['phrase_norm'] == word]
        if not match.empty:
            video_paths.append(match.iloc[0]['vedio'])
    if video_paths:
        return video_paths

    # 3. بحث تقريبي باستخدام rapidfuzz (أقوى من تطابق السياق هنا لأنه أخف وأسرع)
    choices = df_lang['phrase_norm'].tolist()
    best_match, score, idx = process.extractOne(text_norm, choices, scorer=fuzz.ratio)
    if score >= 70:  # عتبة 70% تشابه يمكن تعديلها حسب الحاجة
        return [df_lang.iloc[idx]['vedio']]

    # 4. تطابق سياقي باستخدام embedding
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

# API لاستقبال النص والرد برابط الفيديو
@app.route('/get-video', methods=['POST'])
def get_video():
    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    results = get_video_for_input(user_text, df)
    full_video_urls = [
        url_for('static', filename=f"{path}.mp4", _external=True)
        for path in results
    ]

    return jsonify({"videos": full_video_urls})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

