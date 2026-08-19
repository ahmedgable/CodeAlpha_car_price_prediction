# CarScope — Used Car Price Intelligence

تطبيق Streamlit احترافي لتقدير سعر إعادة بيع السيارات المستعملة اعتمادًا على مجموعة بيانات `car data.csv`، مع واجهة إدخال عربية وصفحة Summary منظمة تعرض صورة مرجعية ومواصفات السيارة والسعر المتوقع ومؤشرات التسعير

# link demo : https://codealphacarpriceprediction-bdvzud5cvwxhkuz8n7frnt.streamlit.app/.

## تشغيل المشروع محليًا

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## ما الذي يقدمه التطبيق؟

تتيح صفحة **التنبؤ بالسعر** اختيار موديل السيارة وتعديل سنة الصنع والسعر الحالي وعدد الكيلومترات ونوع الوقود وناقل الحركة ونوع البائع وعدد المالكين السابقين. بعد الضغط على زر التنبؤ، ينتقل التطبيق تلقائيًا إلى صفحة **ملخص السيارة** التي تعرض الصورة المرجعية المتاحة، السعر المتوقع، النطاق الاسترشادي، مؤشرات الفرق عن السعر الحالي، المواصفات التفصيلية، وسجلات مشابهة من البيانات.

النموذج المستخدم هو `ExtraTreesRegressor` مع نفس منطق الـNotebook الأساسي: إزالة السجلات التي تتجاوز 200,000 كيلومتر، اشتقاق عمر السيارة، ترميز متوسط سعر الموديل، وOne-Hot Encoding للوقود وناقل الحركة ونوع البائع، ثم Standard Scaling قبل التنبؤ.

صور السيارات تُجلب عند الطلب من Wikimedia Commons أو Wikipedia عبر واجهات عامة، مع توضيح المصدر والتنبيه إلى أن الصورة مرجعية وقد لا تكون نفس سنة السيارة أو نفس السيارة الفعلية.

## النشر على Streamlit Community Cloud

1. أنشئ Repository جديدًا على GitHub.
2. ارفع الملفات والمجلدات كما هي، وبالأخص `app.py` و`requirements.txt` و`data/car data.csv`.
3. افتح [share.streamlit.io](https://share.streamlit.io) وسجّل الدخول بحساب GitHub.
4. اختر **Create app**، ثم حدّد الـRepository والفرع والملف الرئيسي `app.py`.
5. اضغط **Deploy** وانتظر اكتمال تثبيت الاعتماديات وتشغيل التطبيق.

لا يحتاج المشروع إلى Secrets أو مفاتيح API. إذا أردت منع جلب الصور الخارجية، يمكن ترك بقية التطبيق تعمل دون تغيير؛ سيعرض التطبيق رسالة عند عدم توفر صورة عامة.

## ملاحظات مهمة

الأسعار المعروضة بوحدة **Lakh** كما في مجموعة البيانات الأصلية. التنبؤ تعليمي واسترشادي وليس تقييمًا رسميًا أو عرض شراء ملزمًا. ملف البيانات هو نسخة CSV عامة من نفس مجموعة البيانات المستخدمة في الـNotebook، مع توحيد أسماء عمودَي `Kms_Driven` و`Seller_Type` إلى الأسماء الموجودة في الـNotebook.
## run project
deactivate
Remove-Item -Recurse -Force .venv
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install --only-binary=:all: -r requirements.txt
streamlit run app.py
