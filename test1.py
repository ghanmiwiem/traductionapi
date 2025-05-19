import requests

# هذا هو الرابط النهائي للـ API الذي شغّلته على Render
url = "https://traductionapi.onrender.com/get-video"

# أدخل أي نص تريد تجربته
data = {"text": "Bonjour"}  # جرّب كلمات مثل: "bonjour", "merci", "baby", إلخ...

# أرسل الطلب
response = requests.post(url, json=data)

# اطبع النتائج
if response.status_code == 200:
    print("✅ الفيديوهات المقترحة:", response.json()["videos"])
else:
    print("❌ خطأ:", response.status_code, response.text)
