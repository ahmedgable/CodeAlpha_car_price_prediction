from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="CarScope | Used Car Price Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "car data.csv"
REFERENCE_YEAR = datetime.now().year


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
:root { --ink:#14213d; --muted:#667085; --line:#e8edf4; --blue:#246bfe; --cyan:#11b6c8; --soft:#f6f8fc; }
html, body, [class*="css"] { font-family: 'Cairo', 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #f8fbff 0%, #ffffff 38%); }
[data-testid="stHeader"] { background: rgba(255,255,255,0.75); }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
.hero { background: radial-gradient(circle at 15% 20%, rgba(17,182,200,.22), transparent 35%), linear-gradient(125deg, #0d1b38 0%, #162d58 48%, #246bfe 100%); color:#fff; border-radius:28px; padding:34px 38px; margin-bottom:24px; box-shadow:0 18px 45px rgba(20,33,61,.15); }
.hero h1 { font-size:clamp(2rem, 4vw, 3.7rem); line-height:1.12; margin:0 0 10px 0; letter-spacing:-1px; color:#fff; }
.hero p { color:rgba(255,255,255,.82); font-size:1.05rem; margin:0; max-width:760px; }
.eyebrow { color:#8ae7ed; text-transform:uppercase; letter-spacing:2px; font-size:.76rem; font-weight:800; margin-bottom:10px; }
.section-title { color:var(--ink); font-size:1.35rem; font-weight:800; margin:22px 0 12px; }
.section-subtitle { color:var(--muted); margin-top:-5px; margin-bottom:18px; }
.card { background:#fff; border:1px solid var(--line); border-radius:20px; padding:20px 22px; box-shadow:0 8px 28px rgba(20,33,61,.055); }
.metric-card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:17px 18px; min-height:100px; box-shadow:0 8px 24px rgba(20,33,61,.045); }
.metric-label { color:var(--muted); font-size:.82rem; margin-bottom:4px; }
.metric-value { color:var(--ink); font-size:1.45rem; font-weight:800; line-height:1.2; }
.metric-note { color:#7b8798; font-size:.73rem; margin-top:4px; }
.price-card { background:linear-gradient(135deg,#0d1b38,#246bfe); color:#fff; border-radius:22px; padding:25px; box-shadow:0 14px 32px rgba(36,107,254,.22); }
.price-card .label { color:rgba(255,255,255,.75); font-size:.86rem; }
.price-card .price { font-size:2.45rem; font-weight:800; line-height:1.15; margin:6px 0; }
.price-card .range { color:#b7f5f7; font-size:.83rem; }
.badge { display:inline-block; padding:6px 11px; border-radius:999px; background:#eaf1ff; color:#246bfe; font-size:.75rem; font-weight:700; margin:2px 4px 2px 0; }
.detail-row { display:flex; justify-content:space-between; gap:18px; border-bottom:1px solid #eef1f5; padding:11px 0; font-size:.92rem; }
.detail-row:last-child { border-bottom:0; }
.detail-key { color:var(--muted); }
.detail-value { color:var(--ink); font-weight:700; text-align:left; }
.image-shell { background:#eef3fb; border-radius:22px; padding:10px; border:1px solid var(--line); }
.source-note { color:#8490a3; font-size:.72rem; margin-top:7px; }
div[data-testid="stSidebar"] { background:#f5f8fd; border-right:1px solid var(--line); }
div[data-testid="stSidebar"] .block-container { padding-top:2rem; }
.stButton > button { border-radius:12px; border:0; background:linear-gradient(135deg,#246bfe,#11b6c8); color:#fff; font-weight:800; min-height:46px; box-shadow:0 8px 18px rgba(36,107,254,.18); }
.stButton > button:hover { border:0; color:#fff; transform:translateY(-1px); }
.stDownloadButton > button { border-radius:12px; font-weight:700; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
hr { border:0; border-top:1px solid var(--line); margin:25px 0; }
.small-muted { color:var(--muted); font-size:.82rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Kms_Driven": "Driven_kms", "Seller_Type": "Selling_type"})
    required = [
        "Car_Name", "Year", "Selling_Price", "Present_Price", "Driven_kms",
        "Fuel_Type", "Selling_type", "Transmission", "Owner",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[df["Driven_kms"] < 200000].copy()


@st.cache_resource(show_spinner="جاري تجهيز نموذج التسعير…")
def build_model():
    raw = load_data()
    work = raw.copy()
    work["age"] = REFERENCE_YEAR - work["Year"]

    global_target = float(work["Selling_Price"].mean())
    target_means = work.groupby("Car_Name")["Selling_Price"].mean().to_dict()
    work["Car_Name_Encoded"] = work["Car_Name"].map(target_means).fillna(global_target)

    onehot_source = work[["Fuel_Type", "Transmission", "Selling_type"]]
    onehot = pd.get_dummies(onehot_source, columns=["Fuel_Type", "Transmission", "Selling_type"], dtype=float)
    onehot_columns = list(onehot.columns)
    numeric = work[["Present_Price", "Driven_kms", "Owner", "age"]].astype(float)
    features = pd.concat([numeric.reset_index(drop=True), onehot.reset_index(drop=True), work[["Car_Name_Encoded"]].reset_index(drop=True)], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        features, work["Selling_Price"], test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = ExtraTreesRegressor(n_estimators=60, max_depth=35, random_state=42, n_jobs=-1)
    model.fit(x_train_scaled, y_train)
    test_predictions = model.predict(x_test_scaled)

    return {
        "raw": raw,
        "model": model,
        "scaler": scaler,
        "target_means": target_means,
        "global_target": global_target,
        "onehot_columns": onehot_columns,
        "feature_columns": list(features.columns),
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "r2": float(r2_score(y_test, test_predictions)),
    }


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_car_image(car_name: str) -> dict:
    aliases = {
        "city": "Honda City",
        "brio": "Honda Brio",
        "jazz": "Honda Jazz",
        "amaze": "Honda Amaze",
        "ciaz": "Maruti Suzuki Ciaz",
        "swift": "Maruti Suzuki Swift",
        "dzire": "Maruti Suzuki Dzire",
        "wagon r": "Maruti Suzuki Wagon R",
        "alto 800": "Maruti Alto 800",
        "alto k10": "Maruti Alto K10",
        "ertiga": "Maruti Suzuki Ertiga",
        "ritz": "Maruti Suzuki Ritz",
        "sx4": "Maruti Suzuki SX4",
        "vitara brezza": "Maruti Suzuki Vitara Brezza",
        "innova": "Toyota Innova",
        "fortuner": "Toyota Fortuner",
        "corolla altis": "Toyota Corolla Altis",
        "verna": "Hyundai Verna",
        "i20": "Hyundai i20",
        "i10": "Hyundai i10",
        "creta": "Hyundai Creta",
        "grand i10": "Hyundai Grand i10",
        "santro": "Hyundai Santro",
        "elantra": "Hyundai Elantra",
        "thar": "Mahindra Thar",
        "xuv500": "Mahindra XUV500",
        "scorpio": "Mahindra Scorpio",
        "bolero": "Mahindra Bolero",
        "hexa": "Tata Hexa",
        "nexon": "Tata Nexon",
        "omni": "Maruti Omni",
        "passion pro": "Hero Passion Pro",
        "splendor": "Hero Splendor",
        "activa 3g": "Honda Activa",
        "activa 4g": "Honda Activa",
        "access 125": "Suzuki Access 125",
        "pulsar 150": "Bajaj Pulsar 150",
        "pulsar 180": "Bajaj Pulsar 180",
        "pulsar ns 200": "Bajaj Pulsar NS200",
        "apache rtr 160": "TVS Apache RTR 160",
        "apache rtr 180": "TVS Apache RTR 180",
        "fzs": "Yamaha FZ-S",
        "fz v 2.0": "Yamaha FZ",
        "r15": "Yamaha YZF-R15",
        "duke 200": "KTM Duke 200",
        "bullet 350": "Royal Enfield Bullet",
        "classic 350": "Royal Enfield Classic",
        "continental gt 650": "Royal Enfield Continental GT",
    }
    query = aliases.get(car_name.lower().strip(), car_name)
    headers = {"User-Agent": "CarScope/1.0 (educational Streamlit app)"}

    try:
        commons_params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} car",
            "gsrnamespace": 6,
            "gsrlimit": 8,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 1000,
            "format": "json",
        }
        response = requests.get("https://commons.wikimedia.org/w/api.php", params=commons_params, headers=headers, timeout=8)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            thumb = info.get("thumburl") or info.get("url")
            title = page.get("title", "")
            if thumb and any(ext in thumb.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
                return {"url": thumb, "page": f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}", "label": "Wikimedia Commons"}
    except requests.RequestException:
        pass

    try:
        wiki_params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 0,
            "gsrlimit": 5,
            "prop": "pageimages|info",
            "piprop": "thumbnail",
            "pithumbsize": 1000,
            "inprop": "url",
            "format": "json",
        }
        response = requests.get("https://en.wikipedia.org/w/api.php", params=wiki_params, headers=headers, timeout=8)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb:
                return {"url": thumb, "page": page.get("fullurl", "https://en.wikipedia.org"), "label": "Wikipedia"}
    except requests.RequestException:
        pass

    return {"url": None, "page": None, "label": None}


def money(value: float) -> str:
    return f"₹ {value:,.2f} Lakh"


def metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def detail_row(label: str, value: str) -> None:
    st.markdown(
        f'<div class="detail-row"><span class="detail-key">{label}</span><span class="detail-value">{value}</span></div>',
        unsafe_allow_html=True,
    )


def create_features(car_name: str, year: int, present_price: float, driven_kms: int, fuel: str, seller: str, transmission: str, owner: int, assets: dict) -> pd.DataFrame:
    age = REFERENCE_YEAR - year
    car_encoded = assets["target_means"].get(car_name, assets["global_target"])
    base = pd.DataFrame([{
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Owner": owner,
        "age": age,
        "Car_Name_Encoded": car_encoded,
    }])
    cats = pd.DataFrame([{"Fuel_Type": fuel, "Transmission": transmission, "Selling_type": seller}])
    onehot = pd.get_dummies(cats, columns=["Fuel_Type", "Transmission", "Selling_type"], dtype=float)
    onehot = onehot.reindex(columns=assets["onehot_columns"], fill_value=0.0)
    features = pd.concat([base[["Present_Price", "Driven_kms", "Owner", "age"]], onehot, base[["Car_Name_Encoded"]]], axis=1)
    return features.reindex(columns=assets["feature_columns"], fill_value=0.0)


def default_profile(raw: pd.DataFrame, car_name: str) -> dict:
    rows = raw[raw["Car_Name"] == car_name]
    if rows.empty:
        rows = raw
    mode = lambda col, fallback: rows[col].mode().iloc[0] if not rows[col].mode().empty else fallback
    return {
        "Year": int(round(rows["Year"].median())),
        "Present_Price": float(rows["Present_Price"].median()),
        "Driven_kms": int(round(rows["Driven_kms"].median())),
        "Fuel_Type": str(mode("Fuel_Type", "Petrol")),
        "Selling_type": str(mode("Selling_type", "Dealer")),
        "Transmission": str(mode("Transmission", "Manual")),
        "Owner": int(round(rows["Owner"].median())),
    }


def render_sidebar(assets: dict) -> str:
    st.sidebar.markdown("## CarScope")
    st.sidebar.caption("Used Car Price Intelligence")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "التنقل",
        ["التنبؤ بالسعر", "ملخص السيارة"],
        index=0 if st.session_state.get("page", "التنبؤ بالسعر") == "التنبؤ بالسعر" else 1,
    )
    if page == "ملخص السيارة" and "result" not in st.session_state:
        st.sidebar.info("نفّذ تنبؤًا أولًا لعرض الملخص الكامل.")
        return "التنبؤ بالسعر"
    st.sidebar.markdown("---")
    st.sidebar.markdown("**عن النموذج**")
    st.sidebar.caption(f"Extra Trees Regressor · R² test = {assets['r2']:.3f}")
    st.sidebar.caption("الأسعار بوحدة Lakh كما في بيانات التدريب.")
    st.sidebar.markdown("---")
    st.sidebar.caption("الصور تُجلب عند الطلب من Wikimedia Commons أو Wikipedia.")
    return page


def render_hero(page: str) -> None:
    if page == "التنبؤ بالسعر":
        title = "اعرف قيمة عربيتك قبل ما تبيع"
        subtitle = "أدخل مواصفات السيارة، وسيقدّم لك CarScope تقديرًا سريعًا ومنظمًا لسعر إعادة البيع المتوقع."
    else:
        title = "ملخص السيارة المختارة"
        subtitle = "كل تفاصيل السيارة، السعر المتوقع، صورة حقيقية، ومؤشرات تساعدك على فهم قرار التسعير."
    st.markdown(
        f'<div class="hero"><div class="eyebrow">CarScope · USED CAR ANALYTICS</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_prediction(assets: dict) -> None:
    raw = assets["raw"]
    st.markdown('<div class="section-title">ابدأ باختيار السيارة</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">يمكنك استخدام القيم الافتراضية من البيانات أو تعديلها لسيناريو مختلف.</div>', unsafe_allow_html=True)

    car_names = sorted(raw["Car_Name"].unique().tolist(), key=lambda x: x.lower())
    selected_car = st.selectbox("موديل السيارة", car_names, index=car_names.index("city") if "city" in car_names else 0)
    profile = default_profile(raw, selected_car)
    widget_key = selected_car.replace(" ", "_").lower()

    left, right = st.columns([1.05, 1.35], gap="large")
    with left:
        rows = raw[raw["Car_Name"] == selected_car]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**لمحة من السوق داخل البيانات**")
        m1, m2 = st.columns(2)
        with m1:
            metric("عدد الإعلانات", f"{len(rows)}", "لنفس الموديل")
        with m2:
            metric("متوسط سعر البيع", money(float(rows["Selling_Price"].mean())), "سعر تاريخي")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<span class='badge'>بيانات تدريب حقيقية</span><span class='badge'>98 موديلًا</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">صورة استرشادية</div>', unsafe_allow_html=True)
        image = get_car_image(selected_car)
        if image["url"]:
            st.image(image["url"], caption=f"{selected_car.title()} · صورة مرجعية", width="stretch")
            st.markdown(f"<div class='source-note'>المصدر: <a href='{image['page']}' target='_blank'>{image['label']}</a>. الصورة تعريفية وقد لا تمثل نفس سنة الصنع.</div>", unsafe_allow_html=True)
        else:
            st.info("لم يتم العثور على صورة عامة مناسبة لهذا الموديل. ستظل كل وظائف التنبؤ متاحة.")

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**مواصفات السيارة**")
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("سنة الصنع", min_value=1990, max_value=REFERENCE_YEAR + 1, value=profile["Year"], step=1, key=f"year_{widget_key}")
            present_price = st.number_input("السعر الحالي الجديد (Lakh)", min_value=0.05, max_value=250.0, value=max(0.05, round(profile["Present_Price"], 2)), step=0.05, key=f"present_{widget_key}")
            driven_kms = st.number_input("عدد الكيلومترات", min_value=0, max_value=1000000, value=profile["Driven_kms"], step=500, key=f"kms_{widget_key}")
            owner = st.selectbox("عدد المالكين السابقين", [0, 1, 2, 3], index=min(profile["Owner"], 3), key=f"owner_{widget_key}")
        with c2:
            fuel = st.selectbox("نوع الوقود", sorted(raw["Fuel_Type"].unique()), index=sorted(raw["Fuel_Type"].unique()).index(profile["Fuel_Type"]), key=f"fuel_{widget_key}")
            seller = st.selectbox("نوع البائع", sorted(raw["Selling_type"].unique()), index=sorted(raw["Selling_type"].unique()).index(profile["Selling_type"]), key=f"seller_{widget_key}")
            transmission = st.selectbox("ناقل الحركة", sorted(raw["Transmission"].unique()), index=sorted(raw["Transmission"].unique()).index(profile["Transmission"]), key=f"trans_{widget_key}")
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.button("احسب السعر واعرض الـ Summary", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        if submit:
            features = create_features(selected_car, int(year), float(present_price), int(driven_kms), fuel, seller, transmission, int(owner), assets)
            scaled = assets["scaler"].transform(features)
            prediction = float(assets["model"].predict(scaled)[0])
            prediction = max(0.05, prediction)
            st.session_state.result = {
                "car_name": selected_car,
                "year": int(year),
                "present_price": float(present_price),
                "driven_kms": int(driven_kms),
                "fuel": fuel,
                "seller": seller,
                "transmission": transmission,
                "owner": int(owner),
                "prediction": prediction,
                "image": get_car_image(selected_car),
            }
            st.session_state.page = "ملخص السيارة"
            st.rerun()


def render_summary(assets: dict) -> None:
    result = st.session_state.get("result")
    if not result:
        st.warning("ارجع إلى صفحة التنبؤ واختر سيارة أولًا.")
        return

    raw = assets["raw"]
    name = result["car_name"]
    image = result.get("image") or get_car_image(name)
    prediction = result["prediction"]
    depreciation = (1 - prediction / result["present_price"]) * 100 if result["present_price"] else 0
    age = REFERENCE_YEAR - result["year"]
    lower = max(0.05, prediction - assets["mae"])
    upper = prediction + assets["mae"]

    image_col, intro_col = st.columns([1.1, 1], gap="large")
    with image_col:
        if image.get("url"):
            st.markdown('<div class="image-shell">', unsafe_allow_html=True)
            st.image(image["url"], caption=f"{name.title()} · صورة مرجعية", width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            if image.get("page"):
                st.markdown(f"<div class='source-note'>المصدر: <a href='{image['page']}' target='_blank'>{image['label']}</a>. الصورة تعريفية وقد تختلف عن السيارة الفعلية.</div>", unsafe_allow_html=True)
        else:
            st.info("تعذر تحميل صورة عامة لهذا الموديل، لكن الملخص والنتيجة متاحان بالكامل.")
    with intro_col:
        st.markdown(f"<div class='eyebrow' style='color:#246bfe'>SELECTED VEHICLE</div><h2 style='color:#14213d;margin-top:0'>{name.title()}</h2>", unsafe_allow_html=True)
        st.markdown(f'<div class="price-card"><div class="label">السعر المتوقع لإعادة البيع</div><div class="price">{money(prediction)}</div><div class="range">النطاق الاسترشادي: {money(lower)} — {money(upper)}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            metric("عمر السيارة", f"{age} سنة", f"حتى {REFERENCE_YEAR}")
        with b:
            metric("دقة النموذج", f"{assets['r2'] * 100:.1f}%", "R² على عينة الاختبار")

    st.markdown('<div class="section-title">مؤشرات التسعير</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        metric("السعر الحالي الجديد", money(result["present_price"]), "Present Price")
    with q2:
        metric("فرق القيمة", money(abs(result["present_price"] - prediction)), "بين السعرين")
    with q3:
        metric("نسبة الانخفاض التقديرية", f"{depreciation:.1f}%", "مقارنة بالسعر الحالي")
    with q4:
        metric("المسافة المقطوعة", f"{result['driven_kms']:,} km", "Driven kilometres")

    details_col, market_col = st.columns([1, 1], gap="large")
    with details_col:
        st.markdown('<div class="section-title">بيانات السيارة</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        detail_row("الموديل", name.title())
        detail_row("سنة الصنع", str(result["year"]))
        detail_row("نوع الوقود", result["fuel"])
        detail_row("ناقل الحركة", result["transmission"])
        detail_row("نوع البائع", result["seller"])
        detail_row("عدد المالكين السابقين", str(result["owner"]))
        detail_row("العمر المحسوب", f"{age} سنة")
        st.markdown('</div>', unsafe_allow_html=True)
    with market_col:
        st.markdown('<div class="section-title">مقارنة مع سيارات مشابهة</div>', unsafe_allow_html=True)
        similar = raw[raw["Car_Name"] == name].copy()
        if similar.empty:
            st.info("لا توجد سجلات مشابهة كافية.")
        else:
            similar_view = similar[["Year", "Selling_Price", "Present_Price", "Driven_kms", "Fuel_Type", "Transmission"]].copy()
            similar_view = similar_view.rename(columns={"Year":"السنة", "Selling_Price":"سعر البيع", "Present_Price":"السعر الحالي", "Driven_kms":"الكيلومترات", "Fuel_Type":"الوقود", "Transmission":"الفتيس"})
            similar_view["سعر البيع"] = similar_view["سعر البيع"].map(lambda x: f"₹ {x:,.2f}")
            similar_view["السعر الحالي"] = similar_view["السعر الحالي"].map(lambda x: f"₹ {x:,.2f}")
            st.dataframe(similar_view.head(8), hide_index=True, width="stretch")

    st.markdown('<div class="section-title">قراءة سريعة للنتيجة</div>', unsafe_allow_html=True)
    if depreciation >= 35:
        insight = "التقدير يشير إلى انخفاض ملحوظ عن السعر الحالي الجديد، وهو أمر متوقع عادةً مع السيارات الأقدم أو الأعلى استخدامًا."
    elif depreciation >= 15:
        insight = "التقدير يعكس انخفاضًا متوسطًا عن السعر الحالي الجديد، مع تأثر النتيجة بالعمر والمسافة المقطوعة والمواصفات."
    else:
        insight = "الفارق بين السعر المتوقع والسعر الحالي الجديد محدود نسبيًا وفق مواصفات الإدخال الحالية."
    st.markdown(f'<div class="card"><strong>الخلاصة:</strong> {insight}<br><span class="small-muted">النتيجة تقديرية وليست تقييمًا رسميًا أو عرض شراء ملزمًا. متوسط الخطأ المطلق للنموذج على الاختبار حوالي {assets["mae"]:.2f} Lakh.</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("تعديل المواصفات وإعادة التنبؤ", width="stretch"):
            st.session_state.page = "التنبؤ بالسعر"
            st.rerun()
    with col_b:
        summary_csv = pd.DataFrame([{
            "Car_Name": name,
            "Year": result["year"],
            "Predicted_Selling_Price_Lakh": round(prediction, 4),
            "Estimated_Lower_Lakh": round(lower, 4),
            "Estimated_Upper_Lakh": round(upper, 4),
            "Present_Price_Lakh": result["present_price"],
            "Driven_kms": result["driven_kms"],
            "Fuel_Type": result["fuel"],
            "Transmission": result["transmission"],
            "Seller_Type": result["seller"],
            "Owner": result["owner"],
        }]).to_csv(index=False).encode("utf-8")
        st.download_button("تحميل ملخص السيارة CSV", data=summary_csv, file_name="car_summary.csv", mime="text/csv", width="stretch")


try:
    assets = build_model()
except Exception as exc:
    st.error(f"حدث خطأ أثناء تجهيز التطبيق: {exc}")
    st.stop()

page = render_sidebar(assets)
render_hero(page)
if page == "التنبؤ بالسعر":
    render_prediction(assets)
else:
    render_summary(assets)

st.markdown("<hr><div class='small-muted'>CarScope يستخدم بيانات سيارات مستعملة لأغراض تعليمية وتحليلية. الأسعار بوحدة Lakh كما وردت في مجموعة البيانات.</div>", unsafe_allow_html=True)
