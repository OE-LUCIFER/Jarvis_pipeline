"""
Multilingual data generator for JARVIS.
Supports Indian and international languages with rich variations.
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from tqdm import tqdm
import itertools

@dataclass
class Language:
    code: str
    name: str
    script: str
    transliteration: bool = False
    # Common words/phrases in this language
    common_words: Dict[str, str] = None
    # Sentence patterns specific to this language
    sentence_patterns: List[str] = None

# Expanded language support
SUPPORTED_LANGUAGES = [
    # Indian Languages
    Language("en", "English", "Latin"),
    Language("hi", "Hindi", "Devanagari"),
    Language("hi-en", "Hinglish", "Latin", True),
    Language("bn", "Bengali", "Bengali"),
    Language("te", "Telugu", "Telugu"),
    Language("ta", "Tamil", "Tamil"),
    Language("mr", "Marathi", "Devanagari"),
    Language("gu", "Gujarati", "Gujarati"),
    Language("kn", "Kannada", "Kannada"),
    Language("ml", "Malayalam", "Malayalam"),
    Language("pa", "Punjabi", "Gurmukhi"),
    Language("or", "Odia", "Odia"),
    Language("as", "Assamese", "Bengali"),
    Language("ks", "Kashmiri", "Arabic"),
    Language("sd", "Sindhi", "Arabic"),
    Language("ur", "Urdu", "Arabic"),
    Language("ne", "Nepali", "Devanagari"),
    Language("si", "Sinhala", "Sinhala"),
    Language("mai", "Maithili", "Devanagari"),
    Language("bho", "Bhojpuri", "Devanagari"),
    Language("raj", "Rajasthani", "Devanagari"),
    Language("sat", "Santali", "Ol Chiki"),
    Language("kok", "Konkani", "Devanagari"),
    Language("doi", "Dogri", "Devanagari"),
    Language("mni", "Manipuri", "Bengali"),
    
    # International Languages
    Language("es", "Spanish", "Latin"),
    Language("fr", "French", "Latin"),
    Language("de", "German", "Latin"),
    Language("it", "Italian", "Latin"),
    Language("pt", "Portuguese", "Latin"),
    Language("ru", "Russian", "Cyrillic"),
    Language("ar", "Arabic", "Arabic"),
    Language("zh", "Chinese", "Chinese"),
    Language("ja", "Japanese", "Japanese"),
    Language("ko", "Korean", "Korean"),
]

# Common translations for each language
LANGUAGE_TRANSLATIONS = {
    "hi": {  # Hindi
        "open": "खोलो",
        "close": "बंद करो",
        "play": "चलाओ",
        "search": "खोजो",
        "find": "ढूंढो",
        "write": "लिखो",
        "create": "बनाओ",
        "show": "दिखाओ",
        "tell": "बताओ",
        "please": "कृपया",
        "thanks": "धन्यवाद",
        "hello": "नमस्ते",
        "bye": "अलविदा",
        "music": "संगीत",
        "video": "वीडियो",
        "image": "चित्र",
        "browser": "ब्राउज़र",
        "volume": "आवाज़",
        "up": "बढ़ाओ",
        "down": "कम करो",
        "reminder": "याद दिलाओ",
        "system": "सिस्टम",
        "content": "सामग्री",
        "google": "गूगल",
        "youtube": "यूट्यूब",
        "exit": "बाहर जाओ",
        "general": "सामान्य",
        "realtime": "वास्तविक समय",
    },
    "bn": {  # Bengali
        "open": "খোলো",
        "close": "বন্ধ করো",
        "play": "চালাও",
        "search": "খোঁজো",
        "find": "খুঁজে দেখো",
        "write": "লেখো",
        "create": "তৈরি করো",
        "show": "দেখাও",
        "tell": "বলো",
        "please": "দয়া করে",
        "thanks": "ধন্যবাদ",
        "hello": "নমস্কার",
        "bye": "বিদায়",
        "music": "সঙ্গীত",
        "video": "ভিডিও",
        "image": "ছবি",
        "browser": "ব্রাউজার",
        "volume": "শব্দ",
        "up": "বাড়াও",
        "down": "কমাও",
        "reminder": "মনে করিয়ে দাও",
        "system": "সিস্টেম",
        "content": "বিষয়বস্তু",
        "google": "গুগল",
        "youtube": "ইউটিউব",
        "exit": "প্রস্থান",
        "general": "সাধারণ",
        "realtime": "রিয়েল টাইম",
    },
    "te": {  # Telugu
        "open": "తెరువు",
        "close": "మూసివేయి",
        "play": "ప్లే చేయి",
        "search": "వెతుకు",
        "find": "కనుగొను",
        "write": "వ్రాయి",
        "create": "సృష్టించు",
        "show": "చూపించు",
        "tell": "చెప్పు",
        "please": "దయచేసి",
        "thanks": "ధన్యవాదాలు",
        "hello": "నమస్కారం",
        "bye": "వీడ్కోలు",
        "music": "సంగీతం",
        "video": "వీడియో",
        "image": "చిత్రం",
        "browser": "బ్రౌజర్",
        "volume": "వాల్యూమ్",
        "up": "పెంచు",
        "down": "తగ్గించు",
        "reminder": "జ్ఞాపిక",
        "system": "సిస్టమ్",
        "content": "కంటెంట్",
        "google": "గూగుల్",
        "youtube": "యూట్యూబ్",
        "exit": "నిష్క్రమించు",
        "general": "సాధారణ",
        "realtime": "రియల్ టైమ్",
    },
    "ta": {  # Tamil
        "open": "திற",
        "close": "மூடு",
        "play": "இயக்கு",
        "search": "தேடு",
        "find": "கண்டுபிடி",
        "write": "எழுது",
        "create": "உருவாக்கு",
        "show": "காட்டு",
        "tell": "சொல்",
        "please": "தயவுசெய்து",
        "thanks": "நன்றி",
        "hello": "வணக்கம்",
        "bye": "பிரியாவிடை",
        "music": "இசை",
        "video": "வீடியோ",
        "image": "படம்",
        "browser": "உலாவி",
        "volume": "ஒலி",
        "up": "அதிகரி",
        "down": "குறை",
        "reminder": "நினைவூட்டல்",
        "system": "கணினி",
        "content": "உள்ளடக்கம்",
        "google": "கூகுள்",
        "youtube": "யூடியூப்",
        "exit": "வெளியேறு",
        "general": "பொது",
        "realtime": "நிகழ்நேரம்",
    },
    "mr": {  # Marathi
        "open": "उघडा",
        "close": "बंद करा",
        "play": "प्ले करा",
        "search": "शोधा",
        "find": "शोधून काढा",
        "write": "लिहा",
        "create": "तयार करा",
        "show": "दाखवा",
        "tell": "सांगा",
        "please": "कृपया",
        "thanks": "धन्यवाद",
        "hello": "नमस्कार",
        "bye": "निरोप",
        "music": "संगीत",
        "video": "व्हिडिओ",
        "image": "चित्र",
        "browser": "ब्राउझर",
        "volume": "आवाज",
        "up": "वाढवा",
        "down": "कमी करा",
        "reminder": "आठवण",
        "system": "सिस्टम",
        "content": "मजकूर",
        "google": "गूगल",
        "youtube": "यूट्यूब",
        "exit": "बाहेर पडा",
        "general": "सामान्य",
        "realtime": "रीअल टाइम",
    },
    "gu": {  # Gujarati
        "open": "ખોલો",
        "close": "બંધ કરો",
        "play": "ચલાવો",
        "search": "શોધો",
        "find": "શોધી કાઢો",
        "write": "લખો",
        "create": "બનાવો",
        "show": "બતાવો",
        "tell": "કહો",
        "please": "કૃપા કરીને",
        "thanks": "આભાર",
        "hello": "નમસ્તે",
        "bye": "આવજો",
        "music": "સંગીત",
        "video": "વિડિઓ",
        "image": "છબી",
        "browser": "બ્રાઉઝર",
        "volume": "અવાજ",
        "up": "વધારો",
        "down": "ઘટાડો",
        "reminder": "યાદ",
        "system": "સિસ્ટમ",
        "content": "સામગ્રી",
        "google": "ગૂગલ",
        "youtube": "યૂટ્યૂબ",
        "exit": "બહાર નીકળો",
        "general": "સામાન્ય",
        "realtime": "રીયલ ટાઈમ",
    },
    "kn": {  # Kannada
        "open": "ತೆರೆ",
        "close": "ಮುಚ್ಚು",
        "play": "ಪ್ಲೇ ಮಾಡು",
        "search": "ಹುಡುಕು",
        "find": "ಕಂಡುಹಿಡಿ",
        "write": "ಬರೆ",
        "create": "ರಚಿಸು",
        "show": "ತೋರಿಸು",
        "tell": "ಹೇಳು",
        "please": "ದಯವಿಟ್ಟು",
        "thanks": "ಧನ್ಯವಾದಗಳು",
        "hello": "ನಮಸ್ಕಾರ",
        "bye": "ಹೋಗಿ ಬರುತ್ತೇನೆ",
        "music": "ಸಂಗೀತ",
        "video": "ವೀಡಿಯೊ",
        "image": "ಚಿತ್ರ",
        "browser": "ಬ್ರೌಸರ್",
        "volume": "ಧ್ವನಿ",
        "up": "ಹೆಚ್ಚಿಸು",
        "down": "ಕಡಿಮೆ ಮಾಡು",
        "reminder": "ನೆನಪು",
        "system": "ಸಿಸ್ಟಮ್",
        "content": "ವಿಷಯ",
        "google": "ಗೂಗಲ್",
        "youtube": "ಯೂಟ್ಯೂಬ್",
        "exit": "ನಿರ್ಗಮಿಸು",
        "general": "ಸಾಮಾನ್ಯ",
        "realtime": "ರಿಯಲ್ ಟೈಮ್",
    },
    "ml": {  # Malayalam
        "open": "തുറക്കുക",
        "close": "അടയ്ക്കുക",
        "play": "പ്ലേ ചെയ്യുക",
        "search": "തിരയുക",
        "find": "കണ്ടെത്തുക",
        "write": "എഴുതുക",
        "create": "സൃഷ്ടിക്കുക",
        "show": "കാണിക്കുക",
        "tell": "പറയുക",
        "please": "ദയവായി",
        "thanks": "നന്ദി",
        "hello": "നമസ്കാരം",
        "bye": "വിട",
        "music": "സംഗീതം",
        "video": "വീഡിയോ",
        "image": "ചിത്രം",
        "browser": "ബ്രൗസർ",
        "volume": "ശബ്ദം",
        "up": "കൂട്ടുക",
        "down": "കുറയ്ക്കുക",
        "reminder": "ഓർമ്മപ്പെടുത്തൽ",
        "system": "സിസ്റ്റം",
        "content": "ഉള്ളടക്കം",
        "google": "ഗൂഗിൾ",
        "youtube": "യൂട്യൂബ്",
        "exit": "പുറത്തുകടക്കുക",
        "general": "പൊതുവായ",
        "realtime": "റിയൽ ടൈം",
    },
    "pa": {  # Punjabi
        "open": "ਖੋਲ੍ਹੋ",
        "close": "ਬੰਦ ਕਰੋ",
        "play": "ਚਲਾਓ",
        "search": "ਖੋਜੋ",
        "find": "ਲੱਭੋ",
        "write": "ਲਿਖੋ",
        "create": "ਬਣਾਓ",
        "show": "ਦਿਖਾਓ",
        "tell": "ਦੱਸੋ",
        "please": "ਕਿਰਪਾ ਕਰਕੇ",
        "thanks": "ਧੰਨਵਾਦ",
        "hello": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
        "bye": "ਅਲਵਿਦਾ",
        "music": "ਸੰਗੀਤ",
        "video": "ਵੀਡੀਓ",
        "image": "ਤਸਵੀਰ",
        "browser": "ਬ੍ਰਾਊਜ਼ਰ",
        "volume": "ਆਵਾਜ਼",
        "up": "ਵਧਾਓ",
        "down": "ਘਟਾਓ",
        "reminder": "ਯਾਦ",
        "system": "ਸਿਸਟਮ",
        "content": "ਸਮੱਗਰੀ",
        "google": "ਗੂਗਲ",
        "youtube": "ਯੂਟਿਊਬ",
        "exit": "ਬਾਹਰ ਜਾਓ",
        "general": "ਆਮ",
        "realtime": "ਰੀਅਲ ਟਾਈਮ",
    },
    "ur": {  # Urdu
        "open": "کھولیں",
        "close": "بند کریں",
        "play": "چلائیں",
        "search": "تلاش کریں",
        "find": "ڈھونڈیں",
        "write": "لکھیں",
        "create": "بنائیں",
        "show": "دکھائیں",
        "tell": "بتائیں",
        "please": "براہ کرم",
        "thanks": "شکریہ",
        "hello": "السلام علیکم",
        "bye": "خدا حافظ",
        "music": "موسیقی",
        "video": "ویڈیو",
        "image": "تصویر",
        "browser": "براؤزر",
        "volume": "آواز",
        "up": "بڑھائیں",
        "down": "کم کریں",
        "reminder": "یاد دہانی",
        "system": "نظام",
        "content": "مواد",
        "google": "گوگل",
        "youtube": "یوٹیوب",
        "exit": "باہر نکلیں",
        "general": "عام",
        "realtime": "حقیقی وقت",
    },
    "es": {  # Spanish
        "open": "abrir",
        "close": "cerrar",
        "play": "reproducir",
        "search": "buscar",
        "find": "encontrar",
        "write": "escribir",
        "create": "crear",
        "show": "mostrar",
        "tell": "decir",
        "please": "por favor",
        "thanks": "gracias",
        "hello": "hola",
        "bye": "adiós",
        "music": "música",
        "video": "video",
        "image": "imagen",
        "browser": "navegador",
        "volume": "volumen",
        "up": "subir",
        "down": "bajar",
        "reminder": "recordatorio",
        "system": "sistema",
        "content": "contenido",
        "google": "google",
        "youtube": "youtube",
        "exit": "salir",
        "general": "general",
        "realtime": "tiempo real",
    },
    "fr": {  # French
        "open": "ouvrir",
        "close": "fermer",
        "play": "jouer",
        "search": "rechercher",
        "find": "trouver",
        "write": "écrire",
        "create": "créer",
        "show": "montrer",
        "tell": "dire",
        "please": "s'il vous plaît",
        "thanks": "merci",
        "hello": "bonjour",
        "bye": "au revoir",
        "music": "musique",
        "video": "vidéo",
        "image": "image",
        "browser": "navigateur",
        "volume": "volume",
        "up": "augmenter",
        "down": "baisser",
        "reminder": "rappel",
        "system": "système",
        "content": "contenu",
        "google": "google",
        "youtube": "youtube",
        "exit": "quitter",
        "general": "général",
        "realtime": "temps réel",
    },
    "de": {  # German
        "open": "öffnen",
        "close": "schließen",
        "play": "abspielen",
        "search": "suchen",
        "find": "finden",
        "write": "schreiben",
        "create": "erstellen",
        "show": "zeigen",
        "tell": "sagen",
        "please": "bitte",
        "thanks": "danke",
        "hello": "hallo",
        "bye": "tschüss",
        "music": "musik",
        "video": "video",
        "image": "bild",
        "browser": "browser",
        "volume": "lautstärke",
        "up": "erhöhen",
        "down": "senken",
        "reminder": "erinnerung",
        "system": "system",
        "content": "inhalt",
        "google": "google",
        "youtube": "youtube",
        "exit": "beenden",
        "general": "allgemein",
        "realtime": "echtzeit",
    },
    "it": {  # Italian
        "open": "aprire",
        "close": "chiudere",
        "play": "riprodurre",
        "search": "cercare",
        "find": "trovare",
        "write": "scrivere",
        "create": "creare",
        "show": "mostrare",
        "tell": "dire",
        "please": "per favore",
        "thanks": "grazie",
        "hello": "ciao",
        "bye": "arrivederci",
        "music": "musica",
        "video": "video",
        "image": "immagine",
        "browser": "browser",
        "volume": "volume",
        "up": "aumentare",
        "down": "diminuire",
        "reminder": "promemoria",
        "system": "sistema",
        "content": "contenuto",
        "google": "google",
        "youtube": "youtube",
        "exit": "uscire",
        "general": "generale",
        "realtime": "tempo reale",
    },
    "pt": {  # Portuguese
        "open": "abrir",
        "close": "fechar",
        "play": "reproduzir",
        "search": "procurar",
        "find": "encontrar",
        "write": "escrever",
        "create": "criar",
        "show": "mostrar",
        "tell": "dizer",
        "please": "por favor",
        "thanks": "obrigado",
        "hello": "olá",
        "bye": "tchau",
        "music": "música",
        "video": "vídeo",
        "image": "imagem",
        "browser": "navegador",
        "volume": "volume",
        "up": "aumentar",
        "down": "diminuir",
        "reminder": "lembrete",
        "system": "sistema",
        "content": "conteúdo",
        "google": "google",
        "youtube": "youtube",
        "exit": "sair",
        "general": "geral",
        "realtime": "tempo real",
    },
    "ru": {  # Russian
        "open": "открыть",
        "close": "закрыть",
        "play": "воспроизвести",
        "search": "искать",
        "find": "найти",
        "write": "написать",
        "create": "создать",
        "show": "показать",
        "tell": "сказать",
        "please": "пожалуйста",
        "thanks": "спасибо",
        "hello": "привет",
        "bye": "пока",
        "music": "музыка",
        "video": "видео",
        "image": "изображение",
        "browser": "браузер",
        "volume": "громкость",
        "up": "увеличить",
        "down": "уменьшить",
        "reminder": "напоминание",
        "system": "система",
        "content": "содержание",
        "google": "гугл",
        "youtube": "ютуб",
        "exit": "выход",
        "general": "общий",
        "realtime": "реальное время",
    },
    "ar": {  # Arabic
        "open": "افتح",
        "close": "أغلق",
        "play": "شغل",
        "search": "ابحث",
        "find": "جد",
        "write": "اكتب",
        "create": "أنشئ",
        "show": "اعرض",
        "tell": "قل",
        "please": "من فضلك",
        "thanks": "شكراً",
        "hello": "مرحباً",
        "bye": "مع السلامة",
        "music": "موسيقى",
        "video": "فيديو",
        "image": "صورة",
        "browser": "متصفح",
        "volume": "الصوت",
        "up": "ارفع",
        "down": "اخفض",
        "reminder": "تذكير",
        "system": "نظام",
        "content": "محتوى",
        "google": "جوجل",
        "youtube": "يوتيوب",
        "exit": "خروج",
        "general": "عام",
        "realtime": "الوقت الحقيقي",
    },
    "zh": {  # Chinese (Simplified)
        "open": "打开",
        "close": "关闭",
        "play": "播放",
        "search": "搜索",
        "find": "查找",
        "write": "写",
        "create": "创建",
        "show": "显示",
        "tell": "告诉",
        "please": "请",
        "thanks": "谢谢",
        "hello": "你好",
        "bye": "再见",
        "music": "音乐",
        "video": "视频",
        "image": "图片",
        "browser": "浏览器",
        "volume": "音量",
        "up": "增加",
        "down": "减少",
        "reminder": "提醒",
        "system": "系统",
        "content": "内容",
        "google": "谷歌",
        "youtube": "油管",
        "exit": "退出",
        "general": "一般",
        "realtime": "实时",
    },
    "ja": {  # Japanese
        "open": "開く",
        "close": "閉じる",
        "play": "再生",
        "search": "検索",
        "find": "探す",
        "write": "書く",
        "create": "作成",
        "show": "表示",
        "tell": "言う",
        "please": "お願いします",
        "thanks": "ありがとう",
        "hello": "こんにちは",
        "bye": "さようなら",
        "music": "音楽",
        "video": "動画",
        "image": "画像",
        "browser": "ブラウザ",
        "volume": "音量",
        "up": "上げる",
        "down": "下げる",
        "reminder": "リマインダー",
        "system": "システム",
        "content": "コンテンツ",
        "google": "グーグル",
        "youtube": "ユーチューブ",
        "exit": "終了",
        "general": "一般",
        "realtime": "リアルタイム",
    },
    "ko": {  # Korean
        "open": "열기",
        "close": "닫기",
        "play": "재생",
        "search": "검색",
        "find": "찾기",
        "write": "쓰기",
        "create": "만들기",
        "show": "보여주기",
        "tell": "말하기",
        "please": "부탁합니다",
        "thanks": "감사합니다",
        "hello": "안녕하세요",
        "bye": "안녕히 가세요",
        "music": "음악",
        "video": "동영상",
        "image": "이미지",
        "browser": "브라우저",
        "volume": "볼륨",
        "up": "올리기",
        "down": "내리기",
        "reminder": "알림",
        "system": "시스템",
        "content": "콘텐츠",
        "google": "구글",
        "youtube": "유튜브",
        "exit": "종료",
        "general": "일반",
        "realtime": "실시간",
    },
    "ne": {  # Nepali
        "open": "खोल्नुहोस्",
        "close": "बंद गर्नुहोस्",
        "play": "बजाउनुहोस्",
        "search": "खोज्नुहोस्",
        "find": "फेला पार्नुहोस्",
        "write": "लेख्नुहोस्",
        "create": "सिर्जना गर्नुहोस्",
        "show": "देखाउनुहोस्",
        "tell": "बताउनुहोस्",
        "please": "कृपया",
        "thanks": "धन्यवाद",
        "hello": "नमस्कार",
        "bye": "बिदा",
        "music": "संगीत",
        "video": "भिडियो",
        "image": "तस्विर",
        "browser": "ब्राउजर",
        "volume": "आवाज",
        "up": "बढाउनुहोस्",
        "down": "घटाउनुहोस्",
        "reminder": "स्मरण",
        "system": "प्रणाली",
        "content": "सामग्री",
        "google": "गुगल",
        "youtube": "युट्युब",
        "exit": "बाहिर निस्कनुहोस्",
        "general": "सामान्य",
        "realtime": "वास्तविक समय",
    },
    "si": {  # Sinhala
        "open": "විවෘත කරන්න",
        "close": "වසන්න",
        "play": "ධාවනය කරන්න",
        "search": "සොයන්න",
        "find": "හොයන්න",
        "write": "ලියන්න",
        "create": "සාදන්න",
        "show": "පෙන්වන්න",
        "tell": "කියන්න",
        "please": "කරුණාකර",
        "thanks": "ස්තූතියි",
        "hello": "ආයුබෝවන්",
        "bye": "ගිහින් එන්නම්",
        "music": "සංගීතය",
        "video": "වීඩියෝ",
        "image": "පින්තූරය",
        "browser": "බ්‍රවුසරය",
        "volume": "ශබ්දය",
        "up": "වැඩි කරන්න",
        "down": "අඩු කරන්න",
        "reminder": "මතක් කිරීම",
        "system": "පද්ධතිය",
        "content": "අන්තර්ගතය",
        "google": "ගූගල්",
        "youtube": "යූටියුබ්",
        "exit": "පිටවන්න",
        "general": "සාමාන්‍ය",
        "realtime": "තත්‍ය කාලීන",
    },
    "as": {  # Assamese
        "open": "খোলক",
        "close": "বন্ধ কৰক",
        "play": "বজাওক",
        "search": "বিচাৰক",
        "find": "বিচাৰি পাওক",
        "write": "লিখক",
        "create": "সৃষ্টি কৰক",
        "show": "দেখুৱাওক",
        "tell": "কওক",
        "please": "অনুগ্ৰহ কৰি",
        "thanks": "ধন্যবাদ",
        "hello": "নমস্কাৰ",
        "bye": "বিদায়",
        "music": "সংগীত",
        "video": "ভিডিও",
        "image": "ছবি",
        "browser": "ব্ৰাউজাৰ",
        "volume": "শব্দ",
        "up": "বঢ়াওক",
        "down": "কমাওক",
        "reminder": "স্মৰণ",
        "system": "চিস্টেম",
        "content": "সমল",
        "google": "গুগল",
        "youtube": "ইউটিউব",
        "exit": "ওলাই যাওক",
        "general": "সাধাৰণ",
        "realtime": "প্ৰকৃত সময়",
    },
    "ks": {  # Kashmiri
        "open": "کھولِو",
        "close": "بند کرِو",
        "play": "چلاؤ",
        "search": "ڳولِو",
        "find": "لبھو",
        "write": "لیکھو",
        "create": "بناؤ",
        "show": "دکھاؤ",
        "tell": "بتاؤ",
        "please": "مہربانی",
        "thanks": "شکریہ",
        "hello": "آداب",
        "bye": "خدا حافظ",
        "music": "موسیقی",
        "video": "ویڈیو",
        "image": "تصویر",
        "browser": "براؤزر",
        "volume": "آواز",
        "up": "بڑھاؤ",
        "down": "گھٹاؤ",
        "reminder": "یاد دہانی",
        "system": "نظام",
        "content": "مواد",
        "google": "گوگل",
        "youtube": "یوٹیوب",
        "exit": "نکلو",
        "general": "عام",
        "realtime": "حقیقی وقت",
    },
    "sd": {  # Sindhi
        "open": "کوليو",
        "close": "بند ڪريو",
        "play": "هلايو",
        "search": "ڳوليو",
        "find": "لھو",
        "write": "لکو",
        "create": "ٺاھيو",
        "show": "ڏيکاريو",
        "tell": "ٻڌايو",
        "please": "مھرباني",
        "thanks": "شڪريو",
        "hello": "سلام",
        "bye": "خدا حافظ",
        "music": "موسيقي",
        "video": "وڊيو",
        "image": "تصوير",
        "browser": "برائوزر",
        "volume": "آواز",
        "up": "وڌايو",
        "down": "گھٽايو",
        "reminder": "ياد",
        "system": "سسٽم",
        "content": "مواد",
        "google": "گوگل",
        "youtube": "يوٽيوب",
        "exit": "نڪرو",
        "general": "عام",
        "realtime": "حقيقي وقت",
    },
    "mai": {  # Maithili
        "open": "खोलू",
        "close": "बंद करू",
        "play": "बजाउ",
        "search": "खोजू",
        "find": "ताकू",
        "write": "लिखू",
        "create": "बनाउ",
        "show": "देखाउ",
        "tell": "कहू",
        "please": "कृपया",
        "thanks": "धन्यवाद",
        "hello": "प्रणाम",
        "bye": "नमस्कार",
        "music": "संगीत",
        "video": "वीडियो",
        "image": "चित्र",
        "browser": "ब्राउजर",
        "volume": "आवाज",
        "up": "बढ़ाउ",
        "down": "घटाउ",
        "reminder": "याद",
        "system": "सिस्टम",
        "content": "सामग्री",
        "google": "गूगल",
        "youtube": "यूट्यूब",
        "exit": "बाहर जाउ",
        "general": "सामान्य",
        "realtime": "वास्तविक समय",
    },
    "bho": {  # Bhojpuri
        "open": "खोलीं",
        "close": "बंद करीं",
        "play": "बजाईं",
        "search": "खोजीं",
        "find": "ढूंढीं",
        "write": "लिखीं",
        "create": "बनाईं",
        "show": "देखाईं",
        "tell": "बताईं",
        "please": "कृपया",
        "thanks": "धन्यवाद",
        "hello": "प्रणाम",
        "bye": "नमस्कार",
        "music": "गाना",
        "video": "वीडियो",
        "image": "फोटो",
        "browser": "ब्राउजर",
        "volume": "आवाज",
        "up": "बढ़ाईं",
        "down": "घटाईं",
        "reminder": "याद",
        "system": "सिस्टम",
        "content": "सामग्री",
        "google": "गूगल",
        "youtube": "यूट्यूब",
        "exit": "बाहर जाईं",
        "general": "सामान्य",
        "realtime": "वास्तविक समय",
    },
    "raj": {  # Rajasthani
        "open": "खोलो",
        "close": "बंद करो",
        "play": "बजाओ",
        "search": "खोजो",
        "find": "ढूंढो",
        "write": "लिखो",
        "create": "बणाओ",
        "show": "दिखाओ",
        "tell": "बताओ",
        "please": "कृपया",
        "thanks": "धन्यवाद",
        "hello": "खम्मा घणी",
        "bye": "राम राम",
        "music": "गाणो",
        "video": "वीडियो",
        "image": "फोटो",
        "browser": "ब्राउजर",
        "volume": "आवाज",
        "up": "बढ़ाओ",
        "down": "घटाओ",
        "reminder": "याद",
        "system": "सिस्टम",
        "content": "सामग्री",
        "google": "गूगल",
        "youtube": "यूट्यूब",
        "exit": "बाहर जाओ",
        "general": "सामान्य",
        "realtime": "वास्तविक समय",
    },
    "sat": {  # Santali (using Ol Chiki script)
        "open": "ᱨᱟᱲᱟ",
        "close": "ᱵᱚᱸᱫᱚ",
        "play": "ᱮᱱᱮᱡ",
        "search": "ᱯᱟᱱᱛᱮ",
        "find": "ᱧᱟᱢ",
        "write": "ᱚᱞ",
        "create": "ᱵᱮᱱᱟᱣ",
        "show": "ᱫᱮᱠᱷᱟᱣ",
        "tell": "ᱢᱮᱱ",
        "please": "ᱫᱟᱭᱟᱠᱟᱛᱮ",
        "thanks": "ᱡᱩᱨᱩᱢ",
        "hello": "ᱡᱚᱦᱟᱨ",
        "bye": "ᱵᱤᱫᱟᱹᱭ",
        "music": "ᱥᱮᱨᱮᱧ",
        "video": "ᱣᱤᱰᱤᱭᱚ",
        "image": "ᱪᱤᱛᱟᱹᱨ",
        "browser": "ᱵᱨᱟᱩᱡᱚᱨ",
        "volume": "ᱨᱟᱲᱟ",
        "up": "ᱪᱮᱛᱟᱱ",
        "down": "ᱞᱟᱛᱟᱨ",
        "reminder": "ᱩᱭᱦᱟᱹᱨ",
        "system": "ᱥᱤᱥᱴᱚᱢ",
        "content": "ᱡᱤᱱᱤᱥ",
        "google": "ᱜᱩᱜᱚᱞ",
        "youtube": "ᱭᱩᱴᱭᱩᱵᱽ",
        "exit": "ᱩᱰᱩᱠ",
        "general": "ᱥᱟᱫᱷᱟᱨᱚᱱ",
        "realtime": "ᱛᱤᱱᱟᱹᱜ ᱚᱠᱛᱚ",
    },
    "kok": {  # Konkani (using Devanagari script)
        "open": "उगडात",
        "close": "बंद करात",
        "play": "खेळयात",
        "search": "सोदात",
        "find": "मेळयात",
        "write": "बरयात",
        "create": "रचात",
        "show": "दाखयात",
        "tell": "सांगात",
        "please": "उपकार करून",
        "thanks": "देवाचे बरें करूं",
        "hello": "नमस्कार",
        "bye": "मुखार मेळटांव",
        "music": "संगीत",
        "video": "व्हिडिओ",
        "image": "चित्र",
        "browser": "ब्राउझर",
        "volume": "आवाज",
        "up": "वयर",
        "down": "सकयल",
        "reminder": "याद",
        "system": "यंत्रणा",
        "content": "विशय",
        "google": "गूगल",
        "youtube": "यूट्यूब",
        "exit": "भायर सरात",
        "general": "सामान्य",
        "realtime": "प्रत्यक्ष वेळ",
    },
    "doi": {  # Dogri (using Devanagari script)
        "open": "खोलो",
        "close": "बंद करो",
        "play": "चलाओ",
        "search": "खोजो",
        "find": "लब्भो",
        "write": "लिखो",
        "create": "बनाओ",
        "show": "दस्सो",
        "tell": "दस्सो",
        "please": "मेहरबानी करी के",
        "thanks": "धन्यवाद",
        "hello": "जै श्री राम",
        "bye": "फेर मिलांगे",
        "music": "संगीत",
        "video": "वीडियो",
        "image": "तस्वीर",
        "browser": "ब्राउजर",
        "volume": "आवाज",
        "up": "उप्पर",
        "down": "थल्ले",
        "reminder": "याद",
        "system": "सिस्टम",
        "content": "सामग्री",
        "google": "गूगल",
        "youtube": "यूट्यूब",
        "exit": "बाहर निकलो",
        "general": "सामान्य",
        "realtime": "असल समा",
    },
    "mni": {  # Manipuri/Meitei (using Bengali script)
        "open": "হাংদোক",
        "close": "থিংজিনবা",
        "play": "শান্নবা",
        "search": "থিবা",
        "find": "ফংবা",
        "write": "ইবা",
        "create": "শেম্বা",
        "show": "উৎপা",
        "tell": "হায়বা",
        "please": "তৌবিয়ু",
        "thanks": "থাগৎচরি",
        "hello": "কর্মনবা",
        "bye": "খুরুম্জরি",
        "music": "ইশৈ",
        "video": "ভিডিও",
        "image": "মিৎ",
        "browser": "ব্রাউজর",
        "volume": "ভোলুম",
        "up": "কাখৎপা",
        "down": "কুম্থবা",
        "reminder": "নিংশিংবা",
        "system": "সিস্তেম",
        "content": "অচনবা",
        "google": "গুগল",
        "youtube": "ইউটিউব",
        "exit": "থাদোকপা",
        "general": "অপুনবা",
        "realtime": "অচৌবা মতম",
    },
    # Add more languages as needed...
}

# Hinglish variations
HINGLISH_VARIATIONS = {
    "open": ["kholo", "start karo", "shuru karo", "launch karo", "run karo"],
    "close": ["band karo", "close karo", "exit karo", "quit karo", "khatam karo"],
    "play": ["chalao", "play karo", "start karo", "shuru karo", "bajao"],
    "search": ["dhundo", "search karo", "khojo", "find karo", "pata karo"],
    "write": ["likho", "type karo", "draft karo", "compose karo", "create karo"],
    "show": ["dikhao", "display karo", "show karo", "dikha do", "dekhao"],
    "tell": ["batao", "bolo", "explain karo", "samjhao", "describe karo"],
    "please": ["plz", "pls", "please", "krpya", "kripya"],
    "thanks": ["shukriya", "dhanyawad", "thanks", "thank u", "thnx"],
    "hello": ["namaste", "hi", "hey", "hello", "hola"],
    "bye": ["alvida", "bye", "tata", "phir milenge", "goodbye"],
}

class MultilingualDataGenerator:
    def __init__(self):
        # Remove MarianMT-related initialization
        self.device = None
        
        # Load common phrases and patterns for each language
        self.load_language_data()
    
    def load_language_data(self):
        """Load language-specific data and patterns."""
        self.language_data = {}
        
        for lang in SUPPORTED_LANGUAGES:
            # Initialize language-specific data structure
            self.language_data[lang.code] = {
                'translations': LANGUAGE_TRANSLATIONS.get(lang.code, {}),
                'patterns': {},
                'variations': [],
                'cultural_phrases': {}
            }
            
            # Add common command patterns
            self.language_data[lang.code]['patterns'] = {
                'open': [
                    '{} करो',
                    '{} खोलो',
                    '{} शुरू करो',
                    'start {}',
                    'launch {}',
                    'run {}'
                ],
                'close': [
                    '{} बंद करो',
                    '{} को बंद करो',
                    'stop {}',
                    'exit {}',
                    'quit {}'
                ],
                'search': [
                    '{} खोजो',
                    '{} सर्च करो',
                    '{} ढूंढो',
                    'find {}',
                    'search for {}',
                    'look up {}'
                ]
            }
            
            # Add Hinglish variations if the language is Hindi
            if lang.code == 'hi':
                self.language_data[lang.code]['variations'].extend([
                    ('open', ['kholo', 'start karo', 'shuru karo', 'launch karo']),
                    ('close', ['band karo', 'stop karo', 'exit karo']),
                    ('search', ['search karo', 'dhundo', 'khojo']),
                    ('play', ['play karo', 'bajao', 'chalao']),
                    ('volume', ['volume', 'awaaz', 'sound']),
                    ('up', ['badhao', 'increase karo', 'up karo']),
                    ('down', ['kam karo', 'decrease karo', 'down karo'])
                ])
            
            # Add cultural-specific greetings and phrases
            self.language_data[lang.code]['cultural_phrases'] = {
                'greetings': self._get_cultural_greetings(lang.code),
                'farewell': self._get_cultural_farewell(lang.code),
                'courtesy': self._get_cultural_courtesy(lang.code)
            }
            
    def _get_cultural_greetings(self, lang_code):
        """Get culture-specific greetings for a language."""
        greetings = {
            'hi': ['नमस्ते', 'नमस्कार', 'प्रणाम', 'जय श्री कृष्णा'],
            'bn': ['নমস্কার', 'আদাব', 'শুভেচ্ছা'],
            'te': ['నమస్కారం', 'వందనాలు', 'శుభోదయం'],
            'ta': ['வணக்கம்', 'நமஸ்காரம்'],
            'mr': ['नमस्कार', 'नमस्ते', 'राम राम'],
            'gu': ['નમસ્તે', 'જય શ્રી કૃષ્ણ', 'કેમ છો'],
            'kn': ['ನಮಸ್ಕಾರ', 'ಶುಭೋದಯ'],
            'ml': ['നമസ്കാരം', 'അഭിവാദ്യങ്ങൾ'],
            'pa': ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਨਮਸਕਾਰ'],
            'or': ['ନମସ୍କାର', 'ଜୟ ଜଗନ୍ନାଥ'],
            'ur': ['السلام علیکم', 'ادآب'],
            'sat': ['ᱡᱚᱦᱟᱨ'],
            'kok': ['नमस्कार', 'देव बरे करुं'],
            'doi': ['जै श्री राम', 'नमस्ते'],
            'mni': ['কর্মনবা'],
            'as': ['নমস্কাৰ', 'আদৰণি'],
            'ks': ['آداب', 'السلام علیکم'],
            'sd': ['سلام', 'ادا'],
            'mai': ['प्रणाम', 'राम राम'],
            'bho': ['प्रणाम', 'राम राम'],
            'raj': ['खम्मा घणी', 'राम राम', 'जय श्री कृष्णा'],
            'ne': ['नमस्कार', 'नमस्ते'],
            'si': ['ආයුබෝවන්', 'සුභ උදෑසනක්'],
            # International languages
            'en': ['hello', 'hi', 'greetings'],
            'es': ['hola', 'buenos días'],
            'fr': ['bonjour', 'salut'],
            'de': ['hallo', 'guten tag'],
            'it': ['ciao', 'buongiorno'],
            'pt': ['olá', 'bom dia'],
            'ru': ['привет', 'здравствуйте'],
            'ar': ['مرحبا', 'السلام عليكم'],
            'zh': ['你好', '早上好'],
            'ja': ['こんにちは', 'おはようございます'],
            'ko': ['안녕하세요', '안녕']
        }
        return greetings.get(lang_code, ['hello'])  # Default to English
        
    def _get_cultural_farewell(self, lang_code):
        """Get culture-specific farewell expressions."""
        farewells = {
            'hi': ['नमस्ते', 'फिर मिलेंगे', 'अलविदा', 'शुभ रात्रि'],
            'bn': ['নমস্কার', 'আবার দেখা হবে', 'বিদায়', 'শুভ রাত্রি'],
            'te': ['నమస్కారం', 'మళ్ళీ కలుద్దాం', 'వీడ్కోలు', 'శుభరాత్రి'],
            'ta': ['வணக்கம்', 'மீண்டும் சந்திப்போம்', 'விடைபெறுகிறேன்', 'இரவு வணக்கம்'],
            'mr': ['नमस्कार', 'पुन्हा भेटू', 'राम राम', 'शुभ रात्री'],
            'gu': ['નમસ્તે', 'ફરી મળીશું', 'આવજો', 'શુભ રાત્રી'],
            'kn': ['ನಮಸ್ಕಾರ', 'ಮತ್ತೆ ಸಿಗೋಣ', 'ಹೋಗಿ ಬರ್ತೀನಿ', 'ಶುಭ ರಾತ್ರಿ'],
            'ml': ['നമസ്കാരം', 'പിന്നെ കാണാം', 'വിട', 'ശുഭ രാത്രി'],
            'pa': ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਫਿਰ ਮਿਲਾਂਗੇ', 'ਅਲਵਿਦਾ', 'ਸ਼ੁਭ ਰਾਤ'],
            'or': ['ନମସ୍କାର', 'ପୁଣି ଦେଖା ହେବ', 'ବିଦାୟ', 'ଶୁଭ ରାତ୍ରି'],
            'ur': ['خدا حافظ', 'پھر ملیں گے', 'الوداع', 'شب بخیر'],
            'sat': ['ᱡᱚᱦᱟᱨ', 'ᱯᱷᱮᱨ ᱧᱟᱢᱚᱜ-ᱟ', 'ᱵᱤᱫᱟᱹᱭ'],
            'kok': ['नमस्कार', 'मुखार मेळटांव', 'देव बरे करुं', 'शुभ रात्र'],
            'doi': ['नमस्ते', 'फेर मिलांगे', 'अलविदा', 'शुभ रात्री'],
            'mni': ['কর্মনবা', 'অমুক পুনা উনসিল্লগা', 'খুরুম্জরি'],
            'as': ['নমস্কাৰ', "আকৌ দেখা হ'ব", 'বিদায়', 'শুভ ৰাতি'],
            'ks': ['خُدا حافِظ', 'پھِر مُلاقات', 'الوداع'],
            'sd': ['خدا حافظ', 'وري ملندا سين', 'الوداع'],
            'mai': ['प्रणाम', 'फेर भेंट होएत', 'विदाइ', 'शुभ रात्रि'],
            'bho': ['प्रणाम', 'फेर मिलब', 'अलविदा', 'शुभ रात्री'],
            'raj': ['राम राम', 'फेर मिलांगे', 'अलविदा', 'शुभ रात्री'],
            'ne': ['नमस्कार', 'फेरी भेटौंला', 'बिदा', 'शुभ रात्री'],
            'si': ['ආයුබෝවන්', 'නැවත හමුවෙමු', 'ගිහින් එන්නම්', 'සුභ රාත්රියක්'],
            # International languages
            'en': ['goodbye', 'bye', 'see you later', 'good night'],
            'es': ['adiós', 'hasta luego', 'hasta pronto', 'buenas noches'],
            'fr': ['au revoir', 'à bientôt', 'adieu', 'bonne nuit'],
            'de': ['auf wiedersehen', 'tschüss', 'bis später', 'gute nacht'],
            'it': ['arrivederci', 'ciao', 'a presto', 'buona notte'],
            'pt': ['adeus', 'até logo', 'até mais', 'boa noite'],
            'ru': ['до свидания', 'пока', 'до встречи', 'спокойной ночи'],
            'ar': ['مع السلامة', 'إلى اللقاء', 'وداعاً', 'تصبح على خير'],
            'zh': ['再见', '拜拜', '回头见', '晚安'],
            'ja': ['さようなら', 'じゃあね', 'また会いましょう', 'おやすみなさい'],
            'ko': ['안녕히 가세요', '안녕', '다음에 봐요', '안녕히 주무세요']
        }
        return farewells.get(lang_code, ['goodbye', 'bye', 'see you later'])
        
    def _get_cultural_courtesy(self, lang_code):
        """Get culture-specific courtesy phrases."""
        courtesy = {
            'hi': {
                'please': ['कृपया', 'दया करके', 'मेहरबानी करके'],
                'thank_you': ['धन्यवाद', 'शुक्रिया', 'आभार'],
                'sorry': ['क्षमा करें', 'माफ़ कीजिये', 'माफ़ करें']
            },
            'bn': {
                'please': ['দয়া করে', 'অনুগ্রহ করে', 'প্লিজ'],
                'thank_you': ['ধন্যবাদ', 'আপনাকে ধন্যবাদ'],
                'sorry': ['ক্ষমা করুন', 'দুঃখিত']
            },
            'te': {
                'please': ['దయచేసి', 'ప్లీజ్'],
                'thank_you': ['ధన్యవాదాలు', 'థాంక్స్'],
                'sorry': ['క్షమించండి', 'సారీ']
            },
            'ta': {
                'please': ['தயவு செய்து', 'ப்ளீஸ்'],
                'thank_you': ['நன்றி', 'மிக்க நன்றி'],
                'sorry': ['மன்னிக்கவும்', 'சாரி']
            },
            'mr': {
                'please': ['कृपया', 'प्लीज'],
                'thank_you': ['धन्यवाद', 'आभारी आहे'],
                'sorry': ['क्षमा करा', 'माफ करा']
            },
            'gu': {
                'please': ['કૃપા કરીને', 'મહેરબાની કરીને'],
                'thank_you': ['આભાર', 'ધન્યવાદ'],
                'sorry': ['માફ કરશો', 'દિલગીર છું']
            },
            'kn': {
                'please': ['ದಯವಿಟ್ಟು', 'ಪ್ಲೀಸ್'],
                'thank_you': ['ಧನ್ಯವಾದಗಳು', 'ಥ್ಯಾಂಕ್ಸ್'],
                'sorry': ['ಕ್ಷಮಿಸಿ', 'ಸಾರಿ']
            },
            'ml': {
                'please': ['ദയവായി', 'പ്ലീസ്'],
                'thank_you': ['നന്ദി', 'വളരെ നന്ദി'],
                'sorry': ['ക്ഷമിക്കണം', 'സോറി']
            },
            'pa': {
                'please': ['ਕਿਰਪਾ ਕਰਕੇ', 'ਮਿਹਰਬਾਨੀ ਕਰਕੇ'],
                'thank_you': ['ਧੰਨਵਾਦ', 'ਸ਼ੁਕਰੀਆ'],
                'sorry': ['ਮਾਫ਼ ਕਰਨਾ', 'ਸੌਰੀ']
            },
            'or': {
                'please': ['ଦୟାକରି', 'ପ୍ଲିଜ୍'],
                'thank_you': ['ଧନ୍ୟବାଦ', 'ବହୁତ ଧନ୍ୟବାଦ'],
                'sorry': ['କ୍ଷମା କରନ୍ତୁ', 'ସାରି']
            },
            'ur': {
                'please': ['براہ کرم', 'مہربانی'],
                'thank_you': ['شکریہ', 'بہت شکریہ'],
                'sorry': ['معذرت', 'معاف کیجیے']
            },
            'sat': {
                'please': ['ᱫᱟᱭᱟᱠᱟᱛᱮ'],
                'thank_you': ['ᱡᱩᱨᱩᱢ'],
                'sorry': ['ᱤᱠᱟᱹ']
            },
            'kok': {
                'please': ['उपकार करून', 'कृपा करून'],
                'thank_you': ['देवाचे बरें करूं', 'उपकार'],
                'sorry': ['माफ करचें', 'माफी मागतां']
            },
            'doi': {
                'please': ['मेहरबानी करी के', 'कृपया'],
                'thank_you': ['धन्यवाद', 'शुक्रिया'],
                'sorry': ['माफ करना', 'सॉरी']
            },
            'mni': {
                'please': ['তৌবিয়ু'],
                'thank_you': ['থাগৎচরি'],
                'sorry': ['শোরি']
            },
            'as': {
                'please': ['অনুগ্রহ কৰি', 'দয়া কৰি'],
                'thank_you': ['ধন্যবাদ', 'বহুত ধন্যবাদ'],
                'sorry': ['ক্ষমা কৰিব', 'দুঃখিত']
            },
            'ks': {
                'please': ['مہربانی کٕرِتھ', 'برائے مہربانی'],
                'thank_you': ['شُکریہ', 'بہ شُکریہ'],
                'sorry': ['معاف کٔرِیو', 'سوری']
            },
            'sd': {
                'please': ['مھرباني ڪري', 'پليز'],
                'thank_you': ['شڪريو', 'مھرباني'],
                'sorry': ['معاف ڪجو', 'سوري']
            },
            'mai': {
                'please': ['कृपया', 'दया कए'],
                'thank_you': ['धन्यवाद', 'बहुत धन्यवाद'],
                'sorry': ['क्षमा करू', 'माफ करू']
            },
            'bho': {
                'please': ['कृपया', 'मेहरबानी'],
                'thank_you': ['धन्यवाद', 'बहुत धन्यवाद'],
                'sorry': ['माफ कीं', 'सॉरी']
            },
            'raj': {
                'please': ['कृपया', 'मेहरबानी करके'],
                'thank_you': ['धन्यवाद', 'आभार'],
                'sorry': ['माफ करो', 'खमा करो']
            },
            'ne': {
                'please': ['कृपया', 'दया गरेर'],
                'thank_you': ['धन्यवाद', 'धेरै धन्यवाद'],
                'sorry': ['माफ गर्नुहोस्', 'क्षमा गर्नुहोस्']
            },
            'si': {
                'please': ['කරුණාකර', 'ප්ලීස්'],
                'thank_you': ['ස්තූතියි', 'බොහොම ස්තූතියි'],
                'sorry': ['සමාවෙන්න', 'සොරි']
            },
            # International languages
            'en': {
                'please': ['please', 'kindly'],
                'thank_you': ['thank you', 'thanks'],
                'sorry': ['sorry', 'I apologize']
            },
            'es': {
                'please': ['por favor', 'por gentileza'],
                'thank_you': ['gracias', 'muchas gracias'],
                'sorry': ['lo siento', 'perdón', 'disculpe']
            },
            'fr': {
                'please': ["s'il vous plaît", "s'il te plaît"],
                'thank_you': ['merci', 'merci beaucoup'],
                'sorry': ['désolé', 'pardon', 'excusez-moi']
            },
            'de': {
                'please': ['bitte', 'bitte schön'],
                'thank_you': ['danke', 'vielen dank'],
                'sorry': ['entschuldigung', 'tut mir leid']
            },
            'it': {
                'please': ['per favore', 'per piacere'],
                'thank_you': ['grazie', 'grazie mille'],
                'sorry': ['scusi', 'mi dispiace']
            },
            'pt': {
                'please': ['por favor', 'por gentileza'],
                'thank_you': ['obrigado', 'muito obrigado'],
                'sorry': ['desculpe', 'sinto muito']
            },
            'ru': {
                'please': ['пожалуйста', 'будьте добры'],
                'thank_you': ['спасибо', 'большое спасибо'],
                'sorry': ['извините', 'простите']
            },
            'ar': {
                'please': ['من فضلك', 'لو سمحت'],
                'thank_you': ['شكراً', 'شكراً جزيلاً'],
                'sorry': ['آسف', 'عذراً', 'المعذرة']
            },
            'zh': {
                'please': ['请', '麻烦'],
                'thank_you': ['谢谢', '感谢'],
                'sorry': ['对不起', '抱歉']
            },
            'ja': {
                'please': ['お願いします', 'どうぞ'],
                'thank_you': ['ありがとう', 'ありがとうございます'],
                'sorry': ['すみません', 'ごめんなさい']
            },
            'ko': {
                'please': ['주세요', '부탁합니다'],
                'thank_you': ['감사합니다', '고맙습니다'],
                'sorry': ['죄송합니다', '미안합니다']
            }
        }
        return courtesy.get(lang_code, {
            'please': ['please'],
            'thank_you': ['thank you'],
            'sorry': ['sorry']
        })
    
    def translate(self, text: str, target_lang: str) -> str:
        """Simple translation using dictionary lookup.
        Falls back to original text if translation not found.
        """
        # Split text into words
        words = text.lower().split()
        
        # Get translations dictionary for target language
        translations = LANGUAGE_TRANSLATIONS.get(target_lang, {})
        
        # Translate each word if possible
        translated_words = [translations.get(word, word) for word in words]
        
        # Join words back together
        return " ".join(translated_words)

    def get_hinglish_variations(self, text: str) -> List[str]:
        """Generate multiple Hinglish variations of the text."""
        words = text.lower().split()
        variations = []
        
        for word in words:
            if word in HINGLISH_VARIATIONS:
                variations.append(HINGLISH_VARIATIONS[word])
            else:
                variations.append([word])
        
        # Generate combinations
        return [" ".join(combo) for combo in itertools.product(*variations)]

    def get_language_variations(self, text: str, lang_code: str) -> List[str]:
        """Generate variations in the target language."""
        if lang_code == "hi-en":
            return self.get_hinglish_variations(text)
        
        translations = LANGUAGE_TRANSLATIONS.get(lang_code, {})
        words = text.lower().split()
        result = []
        
        # Basic translation
        translated = " ".join(translations.get(word, word) for word in words)
        result.append(translated)
        
        # Add variations with honorifics, particles, etc.
        if lang_code == "hi":
            result.extend([
                translated + " जी",
                "कृपया " + translated,
                translated + " प्लीज",
                "जरा " + translated
            ])
        elif lang_code == "bn":
            result.extend([
                translated + " জি",
                "দয়া করে " + translated,
                translated + " প্লিজ",
                "একটু " + translated
            ])
        
        return result

    def generate_template_data(self) -> List[Dict]:
        """Generate base templates for all categories.
        
        Returns:
            List[Dict]: List of template data with categories and patterns
        """
        templates = []
        
        # General query templates
        general_templates = [
            "who was {person}?",
            "how can I {activity}?",
            "what is {topic}?",
            "tell me about {topic}",
            "how does {topic} work?",
            "can you help me with {topic}?",
            "what's the meaning of {word}?",
            "how to {activity}?",
            "{greeting}",
            "{courtesy_phrase}"
        ]
        
        # Realtime query templates
        realtime_templates = [
            "who is {person}?",
            "what's happening in {location}?",
            "tell me news about {topic}",
            "what are the latest updates on {topic}?",
            "what's the current {metric} of {entity}?",
            "how is {person} doing now?",
            "what's new with {topic}?",
            "what's trending in {domain}?"
        ]
        
        # Action templates
        action_templates = {
            "open": [
                "open {app}",
                "launch {app}",
                "start {app}",
                "can you open {app}?",
                "please open {app}"
            ],
            "close": [
                "close {app}",
                "exit {app}",
                "quit {app}",
                "can you close {app}?",
                "please close {app}"
            ],
            "play": [
                "play {song}",
                "play song {song}",
                "can you play {song}?",
                "start playing {song}",
                "play music {song}"
            ],
            "generate_image": [
                "generate image of {image_prompt}",
                "create image of {image_prompt}",
                "make an image of {image_prompt}",
                "generate a picture of {image_prompt}"
            ],
            "reminder": [
                "set a reminder for {datetime} {message}",
                "remind me at {datetime} about {message}",
                "set an alarm for {datetime} {message}",
                "create a reminder for {datetime} {message}"
            ],
            "system": [
                "{system_action} volume",
                "{system_action} the sound",
                "make it {system_action}",
                "{system_action} system volume"
            ],
            "content": [
                "write {content_type} about {topic}",
                "create {content_type} on {topic}",
                "generate {content_type} about {topic}",
                "help me write {content_type} about {topic}"
            ],
            "google_search": [
                "search for {topic} on google",
                "google {topic}",
                "find {topic} on google",
                "search google for {topic}"
            ],
            "youtube_search": [
                "search for {topic} on youtube",
                "find {topic} on youtube",
                "youtube {topic}",
                "look up {topic} on youtube"
            ],
            "exit": [
                "bye",
                "goodbye",
                "see you later",
                "that's all",
                "exit",
                "quit"
            ]
        }
        
        # Add templates with their categories
        for template in general_templates:
            templates.append({
                "template": template,
                "category": "general",
                "requires_translation": True
            })
            
        for template in realtime_templates:
            templates.append({
                "template": template,
                "category": "realtime",
                "requires_translation": True
            })
            
        for action, patterns in action_templates.items():
            for pattern in patterns:
                templates.append({
                    "template": pattern,
                    "category": action,
                    "requires_translation": True if action != "exit" else False
                })
        
        return templates

    def generate_data(
        self,
        num_samples: int = 1000,
        languages: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """Generate multilingual training data.
        
        Args:
            num_samples: Number of samples to generate per language
            languages: List of language codes to generate data for. If None, uses all supported languages
            output_file: Optional file path to save the generated data
            
        Returns:
            List[Dict]: Generated training data with queries and their categories
        """
        if not languages:
            languages = [lang.code for lang in SUPPORTED_LANGUAGES]
            
        training_data = []
        templates = self.generate_template_data()
        
        # Sample data for filling templates
        sample_data = {
            "person": ["Gandhi", "Einstein", "Newton", "Shakespeare", "Leonardo da Vinci"],
            "activity": ["learn programming", "cook pasta", "play guitar", "learn english", "exercise"],
            "topic": ["artificial intelligence", "climate change", "renewable energy", "blockchain", "quantum computing"],
            "word": ["ephemeral", "serendipity", "eloquent", "resilient", "innovative"],
            "location": ["India", "USA", "China", "Russia", "Japan"],
            "metric": ["stock price", "temperature", "population", "gdp", "growth rate"],
            "entity": ["Apple", "Tesla", "Amazon", "Google", "Microsoft"],
            "domain": ["technology", "science", "entertainment", "sports", "business"],
            "app": ["facebook", "whatsapp", "chrome", "notepad", "calculator"],
            "song": ["shape of you", "despacito", "believer", "perfect", "havana"],
            "image_prompt": ["sunset over mountains", "cat playing", "futuristic city", "beautiful garden", "space station"],
            "datetime": ["9:00 AM", "3:30 PM", "tomorrow morning", "next monday", "in 2 hours"],
            "message": ["team meeting", "doctor appointment", "birthday party", "project deadline", "gym session"],
            "system_action": ["increase", "decrease", "mute", "unmute"],
            "content_type": ["email", "report", "essay", "blog post", "letter"]
        }
        
        for lang_code in languages:
            # Load language-specific data
            greetings = self.language_data[lang_code]['cultural_phrases']['greetings']
            courtesy = self.language_data[lang_code]['cultural_phrases']['courtesy']
            
            # Add language-specific samples
            lang_samples = []
            
            for _ in range(num_samples):
                template_data = random.choice(templates)
                template = template_data["template"]
                category = template_data["category"]
                
                # Fill template with random data
                if "{greeting}" in template:
                    query = random.choice(greetings)
                elif "{courtesy_phrase}" in template:
                    courtesy_type = random.choice(list(courtesy.keys()))
                    query = random.choice(courtesy[courtesy_type])
                else:
                    query = template
                    for key in sample_data:
                        if "{" + key + "}" in query:
                            query = query.replace("{" + key + "}", random.choice(sample_data[key]))
                
                # Translate if required
                if template_data["requires_translation"] and lang_code != "en":
                    query = self.translate(query, lang_code)
                
                lang_samples.append({
                    "query": query,
                    "category": category,
                    "language": lang_code
                })
            
            training_data.extend(lang_samples)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        return training_data

# Example usage
if __name__ == "__main__":
    generator = MultilingualDataGenerator()
    data = generator.generate_data(
        num_samples=1000,
        languages=["en", "hi", "hi-en"],
        output_file="data/multilingual_train.json"
    )
    print(f"Generated {len(data)} examples in multiple languages")
