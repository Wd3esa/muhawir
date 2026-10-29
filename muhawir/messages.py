"""Fixed user-facing texts. Answers themselves only come from passages."""

LANGS = ("ar", "en", "ur")
# languages whose answers quote an approved translation of a verse beside it, never one of Muhawir's own
TRANSLATED_LANGS = ("en", "ur")
STYLES = ("kids", "youth", "extended", "newcomer")

TEXT = {
    "ar": {
        "abstain": "لم أجد في المصادر المعتمدة المتاحة لي ما يجيب عن هذا السؤال. "
                   "يمكنك سؤال مختص في العلم الشرعي.",
        "no_reason": {"ayah": "لم يُذكر لهذه الآية ({where}) سبب نزول في مصدر أسباب النزول المعتمد لدي: «{source}».",
                      "surah": "لم يُذكر ل{where} سبب نزول في مصدر أسباب النزول المعتمد لدي: «{source}»."},
        "out_of_scope": "لا أستطيع الحكم في المعاملات المالية المعاصرة، يمكنك مراجعة جهة الفتوى في بلدك.",
        "unavailable": "الخدمة غير متاحة مؤقتًا، فلم أستطع البحث والإجابة الآن. حاول مرة أخرى بعد قليل.",
        "simplify_unavailable": "لم أتمكن هذه المرة من إعداد شرح واضح للصغار مع الحفاظ على دقة المعنى. جرّب إعادة صياغة السؤال.",
        "small_talk": "أهلًا وسهلًا! يسعدني الحديث معك، عمّ تحب أن نتكلم؟",
        "translation_label": "ترجمة لغوية، وليست جوابًا من المصادر.",
        "verse_translation_label": "ترجمة معاني الآية كما هي في الترجمة المعتمدة ({where})، لا ترجمة من مُحاور.",
        "misquoted_verse": "تنبيه: نص الآية في المصحف ({where}): ﴿{verse}﴾، وليس «{quoted}».",
        "verse_not_found": "لا أترجم آيات القرآن بنفسي، بل أنقل ترجمة معانيها المعتمدة. ولم أجد هذا النص آيةً في المصادر المتاحة لي، "
                           "فتأكد من لفظه أو اكتب اسم السورة ورقم الآية.",
        "crisis": "أنا آسف لأنك تمر بهذا، وما تشعر به مهم، وأنت لست وحدك. أرجوك تحدّث الآن مع شخص تثق به: "
                  "أحد أهلك أو صديق قريب. وإن كنت تفكر في إيذاء نفسك أو تشعر أنك في خطر، فاتصل الآن برقم الطوارئ "
                  "في بلدك أو اذهب إلى أقرب مستشفى، أو تواصل مع مختص نفسي. وجودك وحياتك يستحقان العناية.",
        "care_note": "وإن كان هذا السؤال يخصك أو يخص من تحب، فتحدّث الآن مع شخص تثق به، "
                     "وإن كان أحد في خطر فاتصل برقم الطوارئ في بلدك.",
        "comfort": "أنا آسف لما تمر به، وأسأل الله أن يرحم من فقدت وأن يربط على قلبك. إن أحببت أن تحدثني عما تشعر به، "
                   "أو أن أذكر لك ما ورد في الصبر على المصيبة، فأنا هنا معك.",
        "about_muhawir": "أنا مُحاور، أجيب عن أسئلتك من مصادر شرعية معتمدة فقط، وأذكر مع كل جملة مصدرها، ولا أفتي من عندي. "
                         "إذا كتبت نصًا موجودًا بلفظه في المصادر، كآية أو حديث، عرضت لك موضعه وما قبله وما بعده، "
                         "وإن أردت معناه أو حكمه أو صحته فاسألني عنه بما تريد، أو اضغط «اسأل عنه». "
                         "وإذا لم أجد في المصادر جوابًا قلت ذلك بصراحة. ما الذي تريد أن نتحاور فيه؟",
        "offer_declined": "حسنًا، وإن خطر لك سؤال آخر فأنا هنا.",
        "no_question_after_answer": "يبدو أن جوابي لم يكن واضحًا أو لم يكن ما تريده. هل أشرحه لك بطريقة أبسط؟ أو اكتب سؤالك بطريقة أخرى، وسأحاول من جديد.",
        "did_you_mean": "يبدو أن لوحة المفاتيح كانت على الإنجليزية. هل تقصد: «{guess}»؟",
        "no_question": "أنا هنا لأحاورك وأجيبك عن أسئلتك عن الإسلام من المصادر المعتمدة. ما الذي تحب أن تعرفه؟",
        "thanks": "العفو! سعدت بالحديث معك، وإن خطر لك أي سؤال فأنا هنا.",
        "personal_case": "لا أستطيع الإجابة في مثل هذه الحالات، يمكنك مراجعة مختص.",
        "contemporary_quoted": "هذا ما ورد في المصادر من أقوال العلماء، منقولًا عنهم كما ورد:",
        "personal_case_info": "هذه معلومات عامة من المصادر، وليست حكمًا في حالتك:",
        "contemporary_info": "لا أستطيع الحكم في هذه المعاملة بعينها، يمكنك مراجعة جهة الفتوى في بلدك.",
        "judging_people": "لا أحكم على أشخاص أو جماعات بعينهم، فهذا خارج ما أقدّمه. "
                          "يمكنني أن أعرض لك ما تقوله المصادر المعتمدة عن المفاهيم نفسها.",
        "override_attempt": "لا أستطيع تغيير طريقتي: أجيب من المصادر المعتمدة فقط ولا أفتي برأيي. "
                            "إن كان لديك سؤال، فاكتبه وسأبحث عنه في المصادر، أو اسأل مختصًا.",
        "reexplain_lead": "لا بأس، سأشرح لك الجواب نفسه بطريقة أبسط:",
        "partial_note": "المصادر الموجودة تغطي بعض الكلمات",
        "ruling_note": "هذا عرض لأقوال العلماء كما وردت في المصدر، وليس فتوى. وللعمل بمسألة تخصك اسأل مختصًا في العلم الشرعي.",
        "ruling_note_agreed": "هذا ما ورد في المصادر، وليس فتوى. وللعمل بمسألة تخصك اسأل مختصًا في العلم الشرعي.",
        "choose_lang": "يجيب مُحاور حاليًا بالعربية والإنجليزية والأردية فقط. بأي لغة تريد الجواب؟",
        "closest_hadith": "لم أجد هذا الحديث بلفظه في المصادر المتاحة لي. أقرب نص وجدته في {where} "
                          "معروض أدناه، فقارن بينهما؛ فقد لا يكون هو المقصود.{grade} وللحكم على صحة حديث اسأل مختصًا في علم الحديث.",
        "closest_hadith_grade": " وحكم المحدث عليه كما في المصدر: {grade}.",
        "choose_ar": "بالعربية",
        "choose_en": "In English",
        "choose_ur": "اردو میں",
        "translation_pending": "",
        "synthetic": "بيانات تجريبية مصطنعة للاختبار، وليست نصوصًا دينية.",
        "too_long": "السؤال طويل جدًا. اختصره من فضلك.",
        "empty": "اكتب سؤالك أولًا.",
    },
    "en": {
        "abstain": "I could not find anything in the approved sources available to me that answers "
                   "this question. "
                   "You may ask a qualified scholar.",
        "no_reason": {"ayah": "The reasons-of-revelation source I rely on, «{source}», records no reason of revelation for this ayah ({where}).",
                      "surah": "The reasons-of-revelation source I rely on, «{source}», records no reason of revelation for {where}."},
        "out_of_scope": "I can't give rulings on contemporary financial dealings; you can consult the fatwa body in your country.",
        "unavailable": "The service is temporarily unavailable, so I could not search and answer right now. Please try again shortly.",
        "simplify_unavailable": "I could not prepare a clear explanation for children while preserving the meaning this time. Try rephrasing your question.",
        "small_talk": "Hello, and welcome! Glad to talk with you. What would you like to talk about?",
        "translation_label": "A language translation, not an answer from the sources.",
        "verse_translation_label": "The meaning of the verse as given in the approved translation ({where}), not Muhawir's own translation.",
        "misquoted_verse": "Note: the verse reads in the Quran ({where}): ﴿{verse}﴾, not «{quoted}».",
        "verse_not_found": "I do not translate verses of the Quran myself; I quote their approved translation. I could not find this "
                           "text as a verse in my sources, so please check its wording or give the surah and verse number.",
        "crisis": "I am sorry you are going through this. What you feel matters, and you are not alone. Please talk now "
                  "with someone you trust: a family member or a close friend. If you are thinking of hurting yourself or "
                  "feel in danger, call your local emergency number now or go to the nearest hospital, or reach a mental "
                  "health professional. You and your life deserve care.",
        "care_note": "If this question is about you or someone you love, please talk now with someone you trust, "
                     "and if anyone is in danger, call your local emergency number.",
        "comfort": "I am sorry for what you are going through. May God have mercy on the one you lost and give your heart "
                   "strength. If you would like to tell me how you feel, or hear what the sources say about patience in "
                   "hardship, I am here with you.",
        "about_muhawir": "I am Muhawir. I answer from approved Islamic sources only, give the source of every sentence, and "
                         "issue no fatwas of my own. When you write a text found word for word in the sources, such as a verse "
                         "or a hadith, I show where it is and what comes before and after it; if you want its meaning, ruling "
                         "or authenticity, ask me about it or tap «Ask about it». When the sources hold no answer, I say so "
                         "plainly. What would you like to talk about?",
        "offer_declined": "All right. If anything else comes to mind, I'm here.",
        "no_question_after_answer": "It seems my answer was not clear, or not what you wanted. Shall I explain it more simply? Or ask your question another way and I will try again.",
        "did_you_mean": "It looks like the keyboard was set to English. Did you mean «{guess}»?",
        "no_question": "I am here to talk with you and answer your questions about Islam from the approved sources. What would you like to know?",
        "thanks": "You're welcome! I enjoyed talking with you, and I'm here whenever something else comes to mind.",
        "personal_case": "I can't answer cases like this; you can consult a specialist.",
        "contemporary_quoted": "Here is what the scholars said in the sources, quoted as it is:",
        "personal_case_info": "This is general information from the sources, not a ruling on your case:",
        "contemporary_info": "I can't give a ruling on this particular dealing; you can consult the fatwa body in your country.",
        "judging_people": "I do not pass judgement on specific people or groups; that is outside what "
                          "I offer. I can show what the approved sources say about the concepts themselves.",
        "override_attempt": "I cannot change how I work: I answer only from approved sources and do not "
                            "give fatwas of my own. Ask your question and I will look for it in the "
                            "sources, or ask a qualified scholar.",
        "reexplain_lead": "No problem, I will explain the same answer again, more simply:",
        "partial_note": "The available sources cover some of the words",
        "ruling_note": "This presents the scholars' views as the source states them; it is not a fatwa. "
                       "For a matter that concerns you, ask a qualified scholar.",
        "ruling_note_agreed": "This is what the sources state; it is not a fatwa. "
                              "For a matter that concerns you, ask a qualified scholar.",
        "choose_lang": "Muhawir currently answers in Arabic, English and Urdu only. Which language would you like the answer in?",
        "closest_hadith": "I could not find this hadith in these words in the sources available to me. The closest text I "
                          "found, in {where}, is shown below; compare them, as it may not be the one you "
                          "mean.{grade} To know whether a hadith is authentic, ask a specialist in hadith.",
        "closest_hadith_grade": " Its grading as the source gives it: {grade}.",
        "choose_ar": "بالعربية",
        "choose_en": "In English",
        "choose_ur": "اردو میں",
        "translation_pending": "The quotation is shown in its original language. Translation will be "
                               "added once a language model is connected.",
        "synthetic": "Synthetic test data, not religious texts.",
        "too_long": "The question is too long. Please shorten it.",
        "empty": "Please type your question first.",
    },
    "ur": {
        "abstain": "مجھے دستیاب معتبر مصادر میں اس سوال کا جواب نہیں ملا۔ "
                   "آپ کسی مستند عالمِ دین سے پوچھ سکتے ہیں۔",
        "no_reason": {"ayah": "میرے معتبر مصدرِ اسبابِ نزول «{source}» میں اس آیت ({where}) کا کوئی سببِ نزول مذکور نہیں۔",
                      "surah": "میرے معتبر مصدرِ اسبابِ نزول «{source}» میں {where} کا کوئی سببِ نزول مذکور نہیں۔"},
        "out_of_scope": "میں جدید مالی معاملات پر حکم نہیں دے سکتا، آپ اپنے ملک کے دارالافتاء سے رجوع کر سکتے ہیں۔",
        "unavailable": "سروس عارضی طور پر دستیاب نہیں، اس لیے میں ابھی تلاش کر کے جواب نہیں دے سکا۔ تھوڑی دیر بعد دوبارہ کوشش کریں۔",
        "simplify_unavailable": "اس بار معنی کی درستگی برقرار رکھتے ہوئے بچوں کے لیے واضح وضاحت تیار نہیں ہو سکی۔ سوال کو دوسرے الفاظ میں پوچھ کر دیکھیں۔",
        "small_talk": "خوش آمدید! آپ سے بات کر کے خوشی ہوگی، آپ کس بارے میں بات کرنا چاہیں گے؟",
        "translation_label": "یہ لسانی ترجمہ ہے، مصادر سے جواب نہیں۔",
        "verse_translation_label": "آیت کے معانی کا ترجمہ معتبر ترجمے کے مطابق ({where})، محاور کا اپنا ترجمہ نہیں۔",
        "misquoted_verse": "تنبیہ: مصحف میں آیت ({where}) کے الفاظ یہ ہیں: ﴿{verse}﴾، نہ کہ «{quoted}»۔",
        "verse_not_found": "میں قرآن کی آیات کا ترجمہ خود نہیں کرتا، بلکہ ان کے معانی کا معتبر ترجمہ نقل کرتا ہوں۔ مجھے یہ عبارت اپنے مصادر میں "
                           "کسی آیت کے طور پر نہیں ملی، اس لیے اس کے الفاظ دیکھ لیں یا سورت کا نام اور آیت نمبر لکھیں۔",
        "crisis": "مجھے افسوس ہے کہ آپ اس کیفیت سے گزر رہے ہیں۔ آپ جو محسوس کر رہے ہیں وہ اہم ہے، اور آپ اکیلے نہیں ہیں۔ براہِ کرم ابھی کسی "
                  "قابلِ اعتماد شخص سے بات کریں: گھر کے کسی فرد یا قریبی دوست سے۔ اگر آپ خود کو نقصان پہنچانے کا سوچ رہے ہیں یا خطرہ محسوس "
                  "کر رہے ہیں تو ابھی اپنے ملک کے ہنگامی نمبر پر فون کریں یا قریبی ہسپتال جائیں، یا کسی ماہرِ نفسیات سے رابطہ کریں۔ "
                  "آپ اور آپ کی زندگی دیکھ بھال کے مستحق ہیں۔",
        "care_note": "اگر یہ سوال آپ سے یا آپ کے کسی عزیز سے متعلق ہے تو ابھی کسی قابلِ اعتماد شخص سے بات کریں، "
                     "اور اگر کوئی خطرے میں ہو تو اپنے ملک کے ہنگامی نمبر پر فون کریں۔",
        "comfort": "مجھے آپ کے دکھ پر افسوس ہے۔ اللہ آپ کے بچھڑنے والے پر رحم فرمائے اور آپ کے دل کو صبر دے۔ اگر آپ اپنی کیفیت مجھے بتانا چاہیں، "
                   "یا مصیبت پر صبر کے بارے میں مصادر کی باتیں سننا چاہیں، تو میں آپ کے ساتھ ہوں۔",
        "about_muhawir": "میں مُحاور ہوں۔ میں صرف معتبر شرعی مصادر سے جواب دیتا ہوں، ہر جملے کا مصدر بتاتا ہوں اور اپنی طرف سے فتویٰ نہیں دیتا۔ "
                         "اگر آپ کوئی ایسا متن لکھیں جو مصادر میں ہوبہو موجود ہو، جیسے آیت یا حدیث، تو میں اس کا مقام اور اس سے پہلے اور بعد کا متن دکھاتا ہوں؛ "
                         "اس کا معنی، حکم یا صحت جاننی ہو تو اس کے بارے میں پوچھیں یا «اس کے بارے میں پوچھیں» دبائیں۔ "
                         "اگر مصادر میں جواب نہ ملے تو میں صاف بتا دیتا ہوں۔ آپ کس بارے میں بات کرنا چاہیں گے؟",
        "offer_declined": "ٹھیک ہے، اگر کوئی اور سوال ذہن میں آئے تو میں حاضر ہوں۔",
        "no_question_after_answer": "لگتا ہے میرا جواب واضح نہیں تھا یا وہ نہیں تھا جو آپ چاہتے تھے۔ کیا میں اسے آسان انداز میں سمجھاؤں؟ یا اپنا سوال کسی اور طرح لکھیں، میں دوبارہ کوشش کروں گا۔",
        "did_you_mean": "لگتا ہے کی بورڈ انگریزی پر تھا۔ کیا آپ کا مطلب ہے: «{guess}»؟",
        "no_question": "میں یہاں آپ سے بات کرنے اور معتبر مصادر سے اسلام کے بارے میں آپ کے سوالوں کے جواب دینے کے لیے ہوں۔ آپ کیا جاننا چاہتے ہیں؟",
        "thanks": "کوئی بات نہیں! آپ سے بات کر کے خوشی ہوئی، جب بھی کوئی سوال ذہن میں آئے میں حاضر ہوں۔",
        "personal_case": "میں ایسے معاملات میں جواب نہیں دے سکتا، آپ کسی ماہر سے رجوع کر سکتے ہیں۔",
        "contemporary_quoted": "مصادر میں علما کے اقوال میں سے جو ملا، وہ جوں کا توں منقول ہے:",
        "personal_case_info": "یہ مصادر سے عمومی معلومات ہیں، آپ کے معاملے کا حکم نہیں:",
        "contemporary_info": "میں اس خاص معاملے پر حکم نہیں دے سکتا، آپ اپنے ملک کے دارالافتاء سے رجوع کر سکتے ہیں۔",
        "judging_people": "میں متعین افراد یا جماعتوں پر حکم نہیں لگاتا، یہ میرے دائرے سے باہر ہے۔ "
                          "میں آپ کو بتا سکتا ہوں کہ معتبر مصادر خود ان تصورات کے بارے میں کیا کہتے ہیں۔",
        "override_attempt": "میں اپنا طریقہ نہیں بدل سکتا: میں صرف معتبر مصادر سے جواب دیتا ہوں اور اپنی رائے سے فتویٰ نہیں دیتا۔ "
                            "اپنا سوال لکھیے، میں اسے مصادر میں تلاش کروں گا، یا کسی مستند عالم سے پوچھیں۔",
        "reexplain_lead": "کوئی بات نہیں، میں وہی جواب آسان انداز میں دوبارہ سمجھاتا ہوں:",
        "partial_note": "دستیاب مصادر کچھ الفاظ کا احاطہ کرتے ہیں",
        "ruling_note": "یہ علما کے اقوال کا بیان ہے جیسے مصدر میں آئے ہیں، فتویٰ نہیں۔ اپنے کسی معاملے پر عمل کے لیے کسی مستند عالم سے پوچھیں۔",
        "ruling_note_agreed": "یہ وہ ہے جو مصادر میں آیا ہے، فتویٰ نہیں۔ اپنے کسی معاملے پر عمل کے لیے کسی مستند عالم سے پوچھیں۔",
        "choose_lang": "محاور فی الحال صرف عربی، انگریزی اور اردو میں جواب دیتا ہے۔ آپ کس زبان میں جواب چاہتے ہیں؟",
        "closest_hadith": "مجھے یہ حدیث ان الفاظ کے ساتھ اپنے مصادر میں نہیں ملی۔ سب سے قریب عبارت جو {where} میں ملی، نیچے دی گئی ہے؛ "
                          "دونوں کا موازنہ کر لیں، ممکن ہے یہ وہ نہ ہو جو آپ کی مراد ہے۔{grade} کسی حدیث کی صحت جاننے کے لیے علمِ حدیث کے ماہر سے پوچھیں۔",
        "closest_hadith_grade": " مصدر کے مطابق محدث کا حکم: {grade}۔",
        "choose_ar": "بالعربية",
        "choose_en": "In English",
        "choose_ur": "اردو میں",
        "translation_pending": "اقتباس اپنی اصل زبان میں دکھایا گیا ہے۔ لسانی ماڈل جڑنے کے بعد ترجمہ شامل ہوگا۔",
        "synthetic": "جانچ کے لیے مصنوعی ڈیٹا، دینی متون نہیں۔",
        "too_long": "سوال بہت لمبا ہے۔ براہِ کرم اسے مختصر کریں۔",
        "empty": "پہلے اپنا سوال لکھیے۔",
    },
}
