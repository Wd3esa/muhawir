# سجل المكونات والتراخيص

كل مكتبة أو أداة أو نموذج أو بيانات يستعملها الحل، مع ترخيصها وحقوق أصحابها.

| المكوّن | الإصدار | الاستخدام | الترخيص | الرابط |
|---|---|---|---|---|
| FastAPI | 0.142.2 | واجهة HTTP | MIT | https://github.com/fastapi/fastapi |
| Starlette (عبر FastAPI) | 1.0.0 | خادم الويب | BSD-3-Clause | https://github.com/encode/starlette |
| Pydantic (عبر FastAPI) | 2.13.3 | التحقق من المدخلات | MIT | https://github.com/pydantic/pydantic |
| SQLite (عبر مكتبة Python القياسية) | 3.45 | تخزين المقاطع والبحث النصي FTS5 | ملكية عامة (Public Domain) | https://sqlite.org/copyright.html |
| HTTPX | 0.28.1 | الاتصال بواجهة النموذج | BSD-3-Clause | https://github.com/encode/httpx |
| Uvicorn | 0.46.0 | تشغيل الخادم | BSD-3-Clause | https://github.com/encode/uvicorn |
| NumPy | 2.x | البحث بالمتجهات (اختياري، مُطفأ افتراضيًا) | BSD-3-Clause | https://github.com/numpy/numpy |
| pytest (للتطوير) | 9.1.1 | الاختبارات | MIT | https://github.com/pytest-dev/pytest |

التراخيص مأخوذة من بيانات الحزم المثبتة (PyPI metadata).

## خدمات خارجية

| الخدمة | الاستخدام | الشروط |
|---|---|---|
| Ollama Cloud (ollama.com)، النموذج المفتوح gpt-oss:120b | صياغة الجواب من المقاطع، وفهم السؤال، والقراءة الثانية | شروط خدمة Ollama، بمفتاح المشغّل. النموذج gpt-oss من OpenAI بترخيص Apache 2.0 (https://github.com/openai/gpt-oss) |
| Render (render.com)، الخطة المجانية | استضافة رابط التجربة | شروط Render |
| التعرف على الكلام وقراءة النص في المتصفح (Web Speech API) | الإدخال الصوتي وزر «استمع» | خاصية في متصفح المستخدم، وقد يرسل المتصفح الصوت إلى خدمة خارجية، وتنبّه الصفحة إلى ذلك |

يمكن للمشغّل أن يضيف Gemini أو Claude بمفاتيحه (انظر `.env.example`)، بشروط كل مزوّد.

## بيانات مفتوحة

| البيانات | الاستخدام | الترخيص | الرابط |
|---|---|---|---|
| hadith-api (fawazahmed0)، الملفان ara-bukhari وara-muslim | نص صحيح البخاري وصحيح مسلم بالعربية | The Unlicense (ملكية عامة) | https://github.com/fawazahmed0/hadith-api |
| OpenITI (KITAB)، نسخة «بداية المجتهد» Shamela0021739 | نص «بداية المجتهد» لابن رشد (الفقه المقارن) | CC BY-NC-SA 4.0: استعمال غير تجاري مع ذكر المصدر، ويُنشر ما يُشتق من البيانات نفسها بالرخصة ذاتها. يُذكر: Nigst, Romanov, Savant, Seydi, Verkinderen, OpenITI 2025.1.9, DOI 10.5281/zenodo.17767721. لا يُحفظ النص في المستودع، بل يُنزَّل عند بناء البيانات | https://github.com/OpenITI/0600AH |

## ترخيص شيفرة الفريق

لم يُختر ترخيص للشيفرة بعد. حتى يُضاف ملف LICENSE تبقى الحقوق محفوظة افتراضيًا.

## بيانات ونصوص طرف ثالث

نصوص المصادر المعتمدة: تُسجَّل شروط استخدامها في [SOURCES.md](SOURCES.md). لا تُنشر نصوص لا نملك حق نشرها.
