# app.py — Полный Streamlit проект Milk Digitalization
# Требует: pandas, numpy, streamlit, matplotlib, seaborn
# Рекомендуется: scikit-learn (pip install scikit-learn)
# Запуск:  streamlit run app.py

import json
import io
import zipfile
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import sklearn (optional but recommended)
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN = True
except Exception:
    SKLEARN = False

# ---------------------------
# --- Настройки путей ---
# ---------------------------
DATA_DIR = Path(__file__).parent
# fallback path used previously
fallback = Path(r"C:\Users\akenz\OneDrive\Desktop\Управление IT проектов\milk\Milk_Digitalization")
if any(fallback.glob("*.csv")) and not any(DATA_DIR.glob("*.csv")):
    DATA_DIR = fallback

PRODUCTS_CSV = DATA_DIR / "Products.csv"
SAMPLES_CSV = DATA_DIR / "Samples.csv"
MEASUREMENTS_CSV = DATA_DIR / "Measurements.csv"
VITAMINS_CSV = DATA_DIR / "Vitamins_AminoAcids.csv"
STORAGE_CSV = DATA_DIR / "Storage_Conditions.csv"
NORMS_JSON = DATA_DIR / "process_norms.json"

# ---------------------------
# --- Утилиты ---
# ---------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    if not df.empty:
        df.columns = [str(c).strip() for c in df.columns]
    return df

def append_row_csv(path: Path, row: dict, cols_order=None):
    df_new = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    if cols_order:
        for c in cols_order:
            if c not in df_new.columns:
                df_new[c] = ""
        df_new = df_new[cols_order]
    df_new.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')

def parse_numeric(val):
    """Пытается корректно распарсить числа из строк:
       - поддержка запятых, пробелов, ±, экспоненты вида ×10^, x10^, ×10, x10
       - обрезает текст после первого нечислового хвоста
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if s == "" or "не обнаруж" in s.lower():
        return np.nan

    # нормализация формата
    s = s.replace(' ', '')  # "1 234,5" -> "1234,5"
    s = s.replace(',', '.')
    s = s.replace('×10^', 'e').replace('x10^', 'e')
    s = s.replace('×10', 'e').replace('x10', 'e')
    s = s.replace('×', '')  # на всякий случай
    if '±' in s:
        s = s.split('±')[0]

    cleaned = ''
    for ch in s:
        if ch.isdigit() or ch in '.-+eE':
            cleaned += ch
        else:
            break
    try:
        return float(cleaned)
    except Exception:
        return np.nan

def download_zip(paths, filename="Milk_Digitalization_all_csv.zip"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in paths:
            if Path(p).exists():
                z.write(p, arcname=Path(p).name)
    buf.seek(0)
    st.download_button("Скачать ZIP", data=buf, file_name=filename, mime="application/zip")

def embed_pdf(path: Path):
    if not path.exists():
        st.warning("PDF файл не найден.")
        return
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode('utf-8')
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600"></iframe>'
    st.components.v1.html(html, height=600, scrolling=True)

# ---------------------------
# --- Кеш загрузки данных ---
# ---------------------------
@st.cache_data
def load_csvs():
    products = safe_read_csv(PRODUCTS_CSV)
    samples = safe_read_csv(SAMPLES_CSV)
    measurements = safe_read_csv(MEASUREMENTS_CSV)
    vitamins = safe_read_csv(VITAMINS_CSV)
    storage = safe_read_csv(STORAGE_CSV)
    return products, samples, measurements, vitamins, storage

products, samples, measurements, vitamins, storage = load_csvs()

# helper to normalize column names
def ensure_col(df, candidates, new_name):
    if df.empty:
        return df, None
    for col in df.columns:
        for cand in candidates:
            if str(col).strip().lower() == str(cand).strip().lower():
                return df.rename(columns={col: new_name}), new_name
    return df, None

# normalize product columns
products, _ = ensure_col(products, ["product_id","id"], "product_id")
products, _ = ensure_col(products, ["name","product_name","title"], "name")
products, _ = ensure_col(products, ["type","category"], "type")
products, _ = ensure_col(products, ["source"], "source")
products, _ = ensure_col(products, ["description"], "description")

# normalize samples columns
samples, _ = ensure_col(samples, ["sample_id","id"], "sample_id")
samples, _ = ensure_col(samples, ["product_id","product"], "product_id")
samples, _ = ensure_col(samples, ["reg_number"], "reg_number")
samples, _ = ensure_col(samples, ["date_received","date"], "date_received")
samples, _ = ensure_col(samples, ["storage_days","duration_days"], "storage_days")
samples, _ = ensure_col(samples, ["conditions"], "conditions")
samples, _ = ensure_col(samples, ["notes"], "notes")

# normalize measurement columns
measurements, _ = ensure_col(measurements, ["id"], "id")
measurements, _ = ensure_col(measurements, ["sample_id","sample"], "sample_id")
measurements, _ = ensure_col(measurements, ["parameter","param","indicator"], "parameter")
measurements, _ = ensure_col(measurements, ["actual_value","value","measurement"], "actual_value")
measurements, _ = ensure_col(measurements, ["unit"], "unit")
measurements, _ = ensure_col(measurements, ["method"], "method")

# storage
storage, _ = ensure_col(storage, ["sample_id"], "sample_id")
storage, _ = ensure_col(storage, ["temperature_C","temperature_c","temp"], "temperature_C")
storage, _ = ensure_col(storage, ["humidity_pct","humidity"], "humidity_pct")
storage, _ = ensure_col(storage, ["duration_days"], "duration_days")

# to int-like
def to_intlike(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
    return df

products = to_intlike(products, "product_id")
samples = to_intlike(samples, "sample_id")
samples = to_intlike(samples, "product_id")
measurements = to_intlike(measurements, "sample_id")
storage = to_intlike(storage, "sample_id")

# numeric measurements
if 'actual_value' in measurements.columns:
    measurements['actual_numeric'] = measurements['actual_value'].apply(parse_numeric)
else:
    measurements['actual_numeric'] = np.nan

# parse dates
if 'date_received' in samples.columns:
    samples['date_received'] = pd.to_datetime(samples['date_received'], errors='coerce')

# ---------------------------
# --- Нормы (process_norms.json) ---
# ---------------------------
default_norms = {
    "Пастеризация": {"min":72.0, "max":75.0, "unit":"°C", "note":"Типовая пастеризация (72–75°C) — см. протокол."},
    "Охлаждение": {"min":2.0, "max":6.0, "unit":"°C", "note":"Хранение/охлаждение."},
    "Ферментация": {"min":18.0, "max":42.0, "unit":"°C", "note":"Температуры ферментации — зависят от рецептуры."}
}
if NORMS_JSON.exists():
    try:
        norms = json.loads(NORMS_JSON.read_text(encoding='utf-8'))
    except Exception:
        norms = default_norms
else:
    norms = default_norms

# ---------------------------
# --- UI стили и цвета этапов ---
# ---------------------------
st.set_page_config(page_title="Milk Digitalization", layout="wide")
st.markdown("""
<style>
.card{background:#fff;padding:12px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.06);margin-bottom:12px}
.prod-title{font-weight:700;color:#0b4c86}
.step-card{background:#fff;padding:16px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.1);margin:8px 0;border-left:5px solid #0b4c86;transition:all 0.3s ease}
.step-card:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,0.15)}
.step-title{font-weight:600;color:#0b4c86;margin-bottom:8px}
.step-desc{color:#666;font-size:14px}
.arrow{text-align:center;font-size:20px;margin:4px 0;color:#0b4c86}
.step-small{font-size:13px;color:#333}
.table-ok{background:#e6fff2}
.table-bad{background:#fff0f0}
.small-muted{color:#666;font-size:13px}
.footer{color:#888;font-size:12px;margin-top:18px}
.product-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:15px;margin:10px 0;box-shadow:0 8px 25px rgba(0,0,0,0.15)}
.product-card:hover{transform:translateY(-3px);box-shadow:0 12px 35px rgba(0,0,0,0.2)}
</style>
""", unsafe_allow_html=True)

STEP_COLORS = {
    "pasteurization":"#d9534f",
    "cooling":"#0275d8",
    "fermentation":"#5cb85c",
    "accept":"#5bc0de",
    "normalization":"#f0ad4e",
    "homogenization":"#6f42c1",
    "inoculation":"#20c997",
    "coagulation":"#fd7e14",
    "pressing":"#6c757d",
    "filtration":"#007bff",
    "storage":"#17a2b8",
    "final":"#343a40"
}

PRODUCT_COLORS = {
    1: "linear-gradient(135deg,#667eea 0%,#764ba2 100%)",
    2: "linear-gradient(135deg,#f093fb 0%,#f5576c 100%)",
    3: "linear-gradient(135deg,#4facfe 0%,#00f2fe 100%)",
    4: "linear-gradient(135deg,#43e97b 0%,#38f9d7 100%)",
    5: "linear-gradient(135deg,#fa709a 0%,#fee140 100%)"
}

def color_for_step(step_id):
    sid = str(step_id).lower()
    for k,v in STEP_COLORS.items():
        if k in sid:
            return v
    return "#0b4c86"

def color_for_product(product_id):
    return PRODUCT_COLORS.get(product_id, "linear-gradient(135deg,#667eea 0%,#764ba2 100%)")

# ---------------------------
# --- State init & navigation ---
# ---------------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'Главная'
if 'selected_product' not in st.session_state:
    st.session_state['selected_product'] = None
if 'selected_step' not in st.session_state:
    st.session_state['selected_step'] = None
if 'selected_step_label' not in st.session_state:
    st.session_state['selected_step_label'] = None

# Sidebar navigation
st.sidebar.title("Навигация")
nav_choice = st.sidebar.radio(
    "",
    ["Главная", "Продукт", "Модели и аналитика"],
    index=["Главная","Продукт","Модели и аналитика"].index(st.session_state['page'])
    if st.session_state['page'] in ["Главная","Продукт","Модели и аналитика"] else 0
)

# Обновляем состояние только если страница изменилась
if nav_choice != st.session_state['page']:
    st.session_state['page'] = nav_choice
    st.session_state['selected_step'] = None
    st.session_state['selected_step_label'] = None
    st.rerun()

# Provide quick CSV upload area on sidebar
st.sidebar.markdown("---")
st.sidebar.write("Загрузить CSV (опционально)")
u = st.sidebar.file_uploader(
    "Выбери CSV (Products/Samples/Measurements/Vitamins/Storage). Можно по одному.",
    type=["csv"]
)
if u is not None:
    fname = u.name.lower()
    content = u.read()
    if "product" in fname:
        dest = PRODUCTS_CSV
    elif "sample" in fname:
        dest = SAMPLES_CSV
    elif "measure" in fname or "measurement" in fname:
        dest = MEASUREMENTS_CSV
    elif "vitamin" in fname or "amino" in fname:
        dest = VITAMINS_CSV
    elif "storage" in fname:
        dest = STORAGE_CSV
    else:
        dest = None

    if dest:
        try:
            Path(dest).write_bytes(content)
            st.sidebar.success(f"Сохранён {dest.name}")
            st.cache_data.clear()  # очистить кеш загрузки
            products, samples, measurements, vitamins, storage = load_csvs()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Ошибка записи: {e}")
    else:
        st.sidebar.info("Не удалось определить тип файла по имени. Переименуй файл и загрузи снова.")

st.sidebar.markdown("---")
st.sidebar.markdown("Версия: 2.0 — demo")

# Helper to set product and go to product page
def goto_product(pid: int):
    st.session_state['selected_product'] = int(pid)
    st.session_state['page'] = 'Продукт'
    st.session_state['selected_step'] = None
    st.session_state['selected_step_label'] = None
    st.rerun()

# ---------------------------
# --- MAIN: Главная страница ---
# ---------------------------
if st.session_state['page'] == 'Главная':
    st.title("🥛 Milk Digitalization — демо платформа")
    st.markdown("Кратко: платформа для мониторинга партий, визуализации показателей и прототипирования моделей для молокопереработки.")
    st.markdown("---")

    # fixed five products; prefer CSV values if present
    fixed_products = [
        {"product_id":1,"name":"Молоко (коровье)","type":"молоко","source":"коровье","description":"Свежее молоко"},
        {"product_id":2,"name":"Молоко (козье)","type":"молоко","source":"козье","description":"Свежее молоко"},
        {"product_id":3,"name":"Сары ірімшік (коровье)","type":"сыр","source":"коровье","description":"Твёрдый сыр"},
        {"product_id":4,"name":"Сары ірімшік (козье)","type":"сыр","source":"козье","description":"Твёрдый сыр"},
        {"product_id":5,"name":"Айран","type":"кисломолочный","source":"коровье","description":"Кисломолочный продукт"}
    ]

    display_products = []
    for fp in fixed_products:
        chosen = None
        if not products.empty and 'product_id' in products.columns:
            try:
                match = products[products['product_id'] == fp['product_id']]
                if not match.empty:
                    chosen = match.iloc[0].to_dict()
            except Exception:
                chosen = None
        display_products.append(chosen if chosen is not None else fp)

    st.subheader("Наши продукты")
    cols = st.columns(2)
    for i, p in enumerate(display_products):
        with cols[i % 2]:
            product_color = color_for_product(p['product_id'])
            st.markdown(f"""
            <div class="product-card" style="background:{product_color}">
                <div style="font-size:18px;font-weight:bold;margin-bottom:8px">{p['name']}</div>
                <div style="font-size:14px;opacity:0.9">Тип: {p.get('type','-')}</div>
                <div style="font-size:14px;opacity:0.9">Источник: {p.get('source','-')}</div>
                <div style="font-size:13px;margin-top:10px;opacity:0.8">{p.get('description','')}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📊 Открыть детали", key=f"btn_product_{p['product_id']}",
                         use_container_width=True, help="Перейти к журналу партий и измерениям"):
                goto_product(p['product_id'])

    st.markdown("---")
    st.subheader("Быстрые действия")
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("📋 Журнал партий", use_container_width=True):
        st.session_state['page'] = 'Продукт'
        st.rerun()
    if c2.button("📈 Аналитика", use_container_width=True):
        st.session_state['page'] = 'Модели и аналитика'
        st.rerun()
    if c3.button("💾 Скачать CSV ZIP", use_container_width=True):
        download_zip([PRODUCTS_CSV, SAMPLES_CSV, MEASUREMENTS_CSV, VITAMINS_CSV, STORAGE_CSV])

    st.markdown("---")
    with st.expander("📂 Файлы в рабочей папке"):
        files_list = [p.name for p in DATA_DIR.glob("*")
                      if p.suffix.lower() in ['.csv', '.json', '.pdf', '.png', '.jpg', '.jpeg']]
        st.write(files_list)

# ---------------------------
# --- PRODUCT PAGE ---
# ---------------------------
elif st.session_state['page'] == 'Продукт':
    pid = st.session_state.get('selected_product', None)

    if pid is None:
        st.info("Выберите продукт на главной странице.")
        if st.button("Вернуться на главную"):
            st.session_state['page'] = 'Главная'
            st.rerun()
    else:
        # product info
        prod = None
        if not products.empty and 'product_id' in products.columns:
            m = products[products['product_id'] == int(pid)]
            if not m.empty:
                prod = m.iloc[0].to_dict()
        if prod is None:
            names = {1:"Молоко (коровье)",2:"Молоко (козье)",3:"Сары ірімшік (коровье)",4:"Сары ірімшік (козье)",5:"Айран"}
            prod = {"product_id":pid,"name":names.get(pid,f"Продукт {pid}"),"type":"-","source":"-","description":""}

        # header + back
        col1, col2 = st.columns([3,1])
        with col1:
            st.header(prod['name'])
        with col2:
            if st.button("← Назад к продуктам", use_container_width=True):
                st.session_state['page'] = 'Главная'
                st.rerun()

        st.write(f"**Тип:** {prod.get('type','-')}  •  **Источник:** {prod.get('source','-')}")
        if prod.get('description'):
            st.write(prod.get('description'))

        st.markdown("---")
        st.subheader("💡 Процесс изготовления (кликабельная блок-схема)")

        # build product specific steps
        name_low = str(prod['name']).lower()
        if "айран" in name_low:
            steps = [
                ("accept","Приемка и контроль сырья", "📥"),
                ("normalization","Нормализация состава", "⚖️"),
                ("pasteurization","Пастеризация (72–75°C)", "🔥"),
                ("cooling_to_inoc","Охлаждение до заквашивания (~40–42°C)", "❄️"),
                ("inoculation","Добавление закваски", "🧫"),
                ("fermentation","Ферментация (контроль pH)", "⏰"),
                ("final_cooling","Охлаждение и фасовка", "📦")
            ]
        elif "сары" in name_low or "ірімшік" in name_low:
            steps = [
                ("accept","Приемка и подготовка", "📥"),
                ("pasteurization","Пастеризация", "🔥"),
                ("coagulation","Свертывание/коагуляция", "🥛"),
                ("whey_removal","Отделение сыворотки", "💧"),
                ("pressing","Прессование", "⚖️"),
                ("salting","Посолка/обработка", "🧂"),
                ("ripening","Выдержка / созревание", "⏰")
            ]
        else:
            steps = [
                ("accept","Приемка и контроль сырья", "📥"),
                ("filtration","Фильтрация / Нормализация", "⚖️"),
                ("pasteurization","Пастеризация (72–75°C)", "🔥"),
                ("cooling","Охлаждение (2–6°C)", "❄️"),
                ("filling","Розлив / Упаковка", "📦"),
                ("storage","Хранение", "🏪")
            ]

        for idx, (sid, label, icon) in enumerate(steps):
            color = color_for_step(sid)
            st.markdown(f"""
            <div class="step-card" style="border-left-color: {color};">
                <div class="step-title">{icon} {label}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Выбрать этап: {label}", key=f"step_{pid}_{sid}", use_container_width=True):
                st.session_state['selected_step'] = sid
                st.session_state['selected_step_label'] = label
                st.rerun()

            if idx < len(steps) - 1:
                st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)

        # show step detail if selected
        if st.session_state.get('selected_step'):
            st.markdown("---")
            sel = st.session_state['selected_step']
            sel_label = st.session_state.get('selected_step_label', sel)
            st.subheader(f"📋 Детали этапа: {sel_label}")

            # get normative values
            step_norm = None
            try:
                if NORMS_JSON.exists():
                    js = json.loads(NORMS_JSON.read_text(encoding='utf-8'))
                    step_norm = js.get(sel) or js.get(sel_label) or None
            except Exception:
                step_norm = None
            if step_norm is None:
                if "пастер" in sel_label.lower():
                    step_norm = norms.get("Пастеризация")
                elif "охлаж" in sel_label.lower() or "хран" in sel_label.lower():
                    step_norm = norms.get("Охлаждение")
                elif "фермент" in sel_label.lower() or "заквас" in sel_label.lower():
                    step_norm = norms.get("Ферментация")

            if step_norm:
                st.success(f"**Норма:** {step_norm.get('min','-')} — {step_norm.get('max','-')} {step_norm.get('unit','')}")
                if step_norm.get('note'):
                    st.info(step_norm.get('note'))
            else:
                st.warning("Норма для этапа не найдена. Могу сгенерировать process_norms.json из протоколов по запросу.")

            # show samples for this product
            st.write("**📊 Журнал партий (Samples) для продукта:**")
            if 'product_id' in samples.columns:
                prod_samples = samples[samples['product_id'] == int(pid)].copy()
            else:
                prod_samples = pd.DataFrame()
            if prod_samples.empty:
                st.info("Партии для этого продукта отсутствуют. Добавьте партию ниже.")
            else:
                st.dataframe(prod_samples.sort_values(by='date_received', ascending=False).reset_index(drop=True))

            # related measurements & norm check
            st.write("**📈 Измерения (Measurements) для партий продукта:**")
            if 'sample_id' in measurements.columns and not prod_samples.empty:
                rel = measurements[measurements['sample_id'].isin(prod_samples['sample_id'])].copy()
            else:
                rel = pd.DataFrame()
            if rel.empty:
                st.info("Нет измерений для этих партий.")
            else:
                if 'actual_numeric' not in rel.columns and 'actual_value' in rel.columns:
                    rel['actual_numeric'] = rel['actual_value'].apply(parse_numeric)

                # mark temperature rows vs norm if norm present
                if step_norm and 'min' in step_norm and 'max' in step_norm:
                    def check_row(r):
                        pname = str(r.get('parameter','')).lower()
                        if 'темпера' in pname or 'temp' in pname:
                            val = r.get('actual_numeric', np.nan)
                            if pd.isna(val):
                                return "no"
                            if val < step_norm['min'] or val > step_norm['max']:
                                return "bad"
                            return "ok"
                        return "na"
                    rel['status_norm'] = rel.apply(check_row, axis=1)
                    temp_rel = rel[rel['status_norm'] != 'na']
                    other_rel = rel[rel['status_norm'] == 'na']

                    if not temp_rel.empty:
                        bad = temp_rel[temp_rel['status_norm']=='bad']
                        ok = temp_rel[temp_rel['status_norm']=='ok']

                        if not ok.empty:
                            st.success("✅ **Температурные измерения в пределах нормы**")
                            st.dataframe(ok[['sample_id','parameter','actual_value','actual_numeric']].reset_index(drop=True))
                        if not bad.empty:
                            st.error("❌ **Отклонения (вне нормы)**")
                            st.dataframe(bad[['sample_id','parameter','actual_value','actual_numeric']].reset_index(drop=True))

                    if not other_rel.empty:
                        st.info("📋 **Другие измерения:**")
                        st.dataframe(other_rel[['sample_id','parameter','actual_value','actual_numeric']].reset_index(drop=True))
                else:
                    st.dataframe(rel[['sample_id','parameter','actual_value']].reset_index(drop=True))

            # Add Sample form
            st.markdown("### ➕ Добавить новую партию (Sample)")
            with st.form(f"form_add_sample_{pid}", clear_on_submit=True):
                try:
                    new_sample_id = int(samples['sample_id'].max()) + 1 if (
                        'sample_id' in samples.columns and not samples.empty and samples['sample_id'].notna().any()
                    ) else 1
                except Exception:
                    new_sample_id = 1

                col1, col2 = st.columns(2)
                with col1:
                    reg_number = st.text_input("Регистрационный номер", value=f"{200 + new_sample_id}")
                    date_received = st.date_input("Дата поступления", value=datetime.now().date())
                    storage_days = st.number_input("Срок хранения (дни)", min_value=0, value=0)
                with col2:
                    temp_input = st.number_input("Температура (°C)", value=21.0, format="%.2f")
                    humidity = st.number_input("Влажность (%)", value=64)
                    notes = st.text_area("Примечания")

                save_sample = st.form_submit_button("💾 Сохранить партию")

            if save_sample:
                row = {
                    "sample_id": int(new_sample_id),
                    "product_id": int(pid),
                    "reg_number": reg_number,
                    "date_received": date_received.strftime("%Y-%m-%d"),
                    "storage_days": int(storage_days),
                    "conditions": f"{temp_input}°C, {humidity}%",
                    "notes": notes
                }
                try:
                    append_row_csv(SAMPLES_CSV, row, cols_order=["sample_id","product_id","reg_number","date_received","storage_days","conditions","notes"])
                    st.cache_data.clear()
                    products, samples, measurements, vitamins, storage = load_csvs()
                    st.success("✅ Партия добавлена! Обновите страницу чтобы увидеть изменения.")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")

            # Add Measurement form
            st.markdown("### ➕ Добавить измерение (Measurement)")
            with st.form(f"form_add_measurement_{pid}", clear_on_submit=True):
                sample_opts = prod_samples['sample_id'].tolist() if not prod_samples.empty else []
                sample_choice = st.selectbox("Выберите Sample ID", options=sample_opts) if sample_opts else None

                col1, col2 = st.columns(2)
                with col1:
                    param = st.text_input("Параметр (например: pH, Температура, Белок, Жир)")
                    value = st.text_input("Значение (например: 4.6 или 89.54±1.07)")
                with col2:
                    unit = st.text_input("Единица (например: °C, %)", value="")
                    method = st.text_input("Метод (например: ГОСТ...)", value="")

                save_meas = st.form_submit_button("💾 Сохранить измерение")

            if save_meas:
                if sample_choice is None:
                    st.error("❌ Сначала добавьте партию (sample) для этого продукта.")
                else:
                    try:
                        new_mid = int(measurements['id'].max())+1 if (
                            'id' in measurements.columns and not measurements.empty and measurements['id'].notna().any()
                        ) else int(datetime.now().timestamp())
                    except Exception:
                        new_mid = int(datetime.now().timestamp())

                    rowm = {"id": new_mid, "sample_id": int(sample_choice), "parameter": param, "unit": unit, "actual_value": value, "method": method}
                    try:
                        append_row_csv(MEASUREMENTS_CSV, rowm, cols_order=["id","sample_id","parameter","unit","actual_value","method"])
                        st.cache_data.clear()
                        products, samples, measurements, vitamins, storage = load_csvs()
                        st.success("✅ Измерение добавлено! Обновите страницу чтобы увидеть изменения.")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")

# ---------------------------
# --- MODELS & ANALYTICS (Только D1 и D2) ---
# ---------------------------
elif st.session_state['page'] == 'Модели и аналитика':
    st.title("📊 Модели и аналитика — Опыт D1 и D2 (Айран)")
    st.write("Здесь показаны только два эксперимента: D1 (7 суток) и D2 (14 суток). Отображаются вводные таблицы и итоговые графики.")

    # =========================
    # 1) Вводные таблицы
    # =========================
    st.subheader("📄 Вводные данные")
    c1, c2 = st.columns(2)

    # Таблица 4 — D1: Айран, 7 суток
    data_D1 = {
        "Группа": ["Контроль", "Опыт 1 (добавка 1)", "Опыт 2 (добавка 2)"],
        "pH": [3.69, 3.65, 3.51],
        "°T": [91, 92, 97],
        "LAB (КОЕ/см³)": [1.2e6, 1.6e6, 2.1e6],
    }
    df_D1 = pd.DataFrame(data_D1)
    df_D1["log10(LAB)"] = np.log10(df_D1["LAB (КОЕ/см³)"].astype(float))

    with c1:
        st.markdown("**Таблица 4. D1 — Айран (7 суток)**")
        st.dataframe(df_D1, use_container_width=True)

    # Таблица 5 — D2: Айран, 14 суток
    data_D2 = {
        "Группа": ["Контроль", "Опыт 1", "Опыт 2"],
        "Белок %": [1.96, 2.05, 2.23],
        "Углеводы %": [2.73, 3.06, 3.85],
        "Жир %": [2.05, 1.93, 2.71],
        "Влага %": [92.56, 92.26, 90.40],
        "АОА вод. (мг/г)": [0.10, 0.15, 0.12],
        "АОА жир (мг/г)": [0.031, 0.043, 0.041],
        "VitC (мг/100г)": [0.880, 0.904, 0.897],
    }
    df_D2 = pd.DataFrame(data_D2)

    with c2:
        st.markdown("**Таблица 5. D2 — Айран (14 суток)**")
        st.dataframe(df_D2, use_container_width=True)

    st.markdown("---")

    # =========================
    # 2) Итоговые графики
    # =========================
    st.subheader("📈 Итоговые графики")

    tab1, tab2, tab3 = st.tabs(["D1: кислотность и LAB", "D2: состав и свойства", "Моделирование pH"])

    # -------- TAB 1: D1 --------
    with tab1:
        # Гистограмма pH + линия log10(LAB)
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.bar(df_D1["Группа"], df_D1["pH"])
        ax1.set_ylabel("pH")
        ax1.set_title("D1 (7 суток): кислотность и рост LAB")

        ax2 = ax1.twinx()
        ax2.plot(df_D1["Группа"], df_D1["log10(LAB)"], marker="o", linewidth=2)
        ax2.set_ylabel("log10(LAB)")

        st.pyplot(fig, use_container_width=True)

    # -------- TAB 2: D2 --------
    with tab2:
        # Состав (Белок/Углеводы/Жир)
        df_comp = df_D2.melt(id_vars="Группа",
                             value_vars=["Белок %", "Углеводы %", "Жир %"],
                             var_name="Показатель", value_name="Значение")

        fig1, ax = plt.subplots(figsize=(8,5))
        # сгруппированные столбцы
        groups = df_comp["Группа"].unique()
        cats = df_comp["Показатель"].unique()
        x = np.arange(len(groups))
        width = 0.8 / len(cats)

        for i, cat in enumerate(cats):
            vals = df_comp[df_comp["Показатель"] == cat]["Значение"].values
            ax.bar(x + i*width - (len(cats)-1)*width/2, vals, width=width, label=cat)

        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("Процент содержания (%)")
        ax.set_title("D2 (14 суток): состав айрана")
        ax.legend()
        st.pyplot(fig1, use_container_width=True)

        # АОА (водная фаза) и витамин C
        fig2, axes = plt.subplots(1, 2, figsize=(12,5))
        axes[0].bar(df_D2["Группа"], df_D2["АОА вод. (мг/г)"])
        axes[0].set_title("АОА (водная фаза)")
        axes[0].set_ylabel("АОА, мг/г")

        axes[1].bar(df_D2["Группа"], df_D2["VitC (мг/100г)"])
        axes[1].set_title("Витамин C")
        axes[1].set_ylabel("VitC, мг/100г")

        plt.suptitle("D2: функциональные свойства", fontsize=14)
        st.pyplot(fig2, use_container_width=True)

    # -------- TAB 3: Моделирование pH --------
    with tab3:
        # Экспериментальные данные времени/ pH
        time = np.array([2, 4, 6, 8, 10])
        ph_control = np.array([4.515, 4.433, 4.386, 4.352, 4.325])
        ph_exp1 = np.array([4.464, 4.394, 4.352, 4.323, 4.300])
        ph_exp2 = np.array([4.419, 4.333, 4.282, 4.246, 4.218])

        st.markdown("**Динамика pH (контроль, опыт 1, опыт 2)**")
        fig0, ax0 = plt.subplots(figsize=(8,5))
        ax0.plot(time, ph_control, 'o-', label='Контроль')
        ax0.plot(time, ph_exp1, 's-', label='Опыт 1')
        ax0.plot(time, ph_exp2, '^-', label='Опыт 2')
        ax0.set_xlabel('Время ферментации, ч'); ax0.set_ylabel('pH')
        ax0.set_title('Сравнение динамики pH (2–10 ч)')
        ax0.grid(True, alpha=0.3); ax0.legend()
        st.pyplot(fig0, use_container_width=True)

        st.markdown("**Модели для pH(t): логарифмическая и гиперболическая (без SciPy)**")

        # Примерные лабораторные данные (как в твоём коде для подгонки)
        t_fit = np.array([1, 2, 3, 4, 5, 6, 8, 10], dtype=float)
        pH_exp = np.array([4.65, 4.50, 4.33, 4.20, 4.05, 3.90, 3.78, 3.70], dtype=float)

        # --- Логарифмическая модель: y = α - β ln(t)
        # линейная регрессия по признаку ln(t): y = c0 + c1*ln(t) => α=c0, β=-c1
        ln_t = np.log(t_fit)
        c1, c0 = np.polyfit(ln_t, pH_exp, 1)  # y = c1*ln(t) + c0
        alpha = c0
        beta = -c1

        # --- Гиперболическая модель: y = a + b/t
        inv_t = 1.0 / t_fit
        m, a_intercept = np.polyfit(inv_t, pH_exp, 1)  # y = m*(1/t) + a_intercept
        a = a_intercept
        b = m

        # Прогнозные кривые
        t_pred = np.linspace(1, 10, 100)
        pH_log_pred = alpha - beta * np.log(t_pred)
        pH_inv_pred = a + b / t_pred

        # Визуализация подгонки
        fig1, ax1 = plt.subplots(figsize=(8,5))
        ax1.scatter(t_fit, pH_exp, color='black', label='Экспериментальные точки')
        ax1.plot(t_pred, pH_log_pred, label='Логарифмическая модель  pH = α - β ln(t)')
        ax1.plot(t_pred, pH_inv_pred, linestyle='--', label='Гиперболическая модель  pH = a + b/t')
        ax1.set_xlabel('Время ферментации, ч'); ax1.set_ylabel('pH')
        ax1.set_title('Моделирование динамики pH при ферментации айрана')
        ax1.grid(True, alpha=0.3); ax1.legend()
        st.pyplot(fig1, use_container_width=True)

        st.markdown("**Оценённые параметры моделей:**")
        st.code(
            f"Логарифмическая:  pH(t) = {alpha:.3f} - {beta:.3f} · ln(t)\n"
            f"Гиперболическая:  pH(t) = {a:.3f} + {b:.3f} / t",
            language="text"
        )

        # Доп. графики по опытам
        st.markdown("**Дополнительные графики:**")

        # Опыт 1: кривая pH
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(time, ph_exp1, 'o-', label='Опыт 1 (модель)')
        ax2.set_xlabel('Время, ч'); ax2.set_ylabel('Прогнозируемый pH')
        ax2.set_title('Опыт 1: динамика pH')
        ax2.grid(True, alpha=0.3); ax2.legend()
        st.pyplot(fig2, use_container_width=True)

        # Опыт 1: поверхность отклика pH(t, dose)
        # pH = 4.535 - 0.102 ln(t) - 0.02 * dose
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        tgrid = np.linspace(2, 10, 30)
        dose = np.linspace(0, 3, 30)
        T, D = np.meshgrid(tgrid, dose)
        pH_surface_exp1 = 4.535 - 0.102 * np.log(T) - 0.02 * D

        fig3 = plt.figure(figsize=(6,4))
        ax3 = fig3.add_subplot(111, projection='3d')
        surf = ax3.plot_surface(D, T, pH_surface_exp1, cmap='autumn')
        ax3.set_xlabel('Доза добавки 1, %')
        ax3.set_ylabel('Время, ч')
        ax3.set_zlabel('pH')
        ax3.set_title('Опыт 1: поверхность отклика pH(t, доза)')
        fig3.colorbar(surf, shrink=0.6, aspect=10)
        st.pyplot(fig3, use_container_width=True)

        # Опыт 2: кривая pH
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.plot(time, ph_exp2, 'o-', label='Опыт 2 (модель)')
        ax4.set_xlabel('Время, ч'); ax4.set_ylabel('Прогнозируемый pH')
        ax4.set_title('Опыт 2: динамика pH')
        ax4.grid(True, alpha=0.3); ax4.legend()
        st.pyplot(fig4, use_container_width=True)

        # Опыт 2: обратная зависимость (pH -> время)
        fig5, ax5 = plt.subplots(figsize=(6,4))
        ax5.plot(ph_exp2, time, 's-')
        ax5.set_xlabel('pH'); ax5.set_ylabel('Время ферментации, ч')
        ax5.set_title('Опыт 2: обратная зависимость (pH → t)')
        ax5.grid(True, alpha=0.3)
        st.pyplot(fig5, use_container_width=True)

        # Сравнение контроль / опыт 1 / опыт 2 — вместе
        fig6, ax6 = plt.subplots(figsize=(7,5))
        ax6.plot(time, ph_control, 'o-', label='Контроль')
        ax6.plot(time, ph_exp1, 's-', label='Опыт 1')
        ax6.plot(time, ph_exp2, '^-', label='Опыт 2')
        ax6.set_xlabel('Время ферментации, ч'); ax6.set_ylabel('pH')
        ax6.set_title('Сравнение: Контроль vs Опыт 1 vs Опыт 2')
        ax6.grid(True, alpha=0.3); ax6.legend()
        st.pyplot(fig6, use_container_width=True)

        st.markdown("**Краткая интерпретация:**")
        st.write("- Снижение pH и рост LAB указывают на активное брожение; максимальный LAB — в опыте 2 (D1).")
        st.write("- На 14-е сутки (D2) повышаются белок и углеводы; баланс жира/влаги зависит от добавок.")
        st.write("- Оценённые модели pH(t) (логарифм/гипербола) демонстрируют замедление снижения pH к стационарной стадии.")

# ---------------------------
# --- Footer ---
# ---------------------------
st.markdown("---")
st.markdown("""
<div class='footer'>
    <div style='text-align: center; padding: 20px;'>
        <h3>🥛 Milk Digitalization Platform</h3>
        <p>Версия 2.0 | Разработано для автоматизации молокоперерабатывающего производства</p>
        <p>📧 Поддержка: demo@milk-digitalization.kz | 📞 +7 (777) 123-45-67</p>
        <div style='margin-top: 15px;'>
            <small>Возможности платформы: мониторинг партий, контроль качества, аналитика, прогнозирование</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Кнопка перезагрузки данных в футере
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🔄 Обновить данные и перезагрузить страницу", use_container_width=True):
        st.cache_data.clear()
        products, samples, measurements, vitamins, storage = load_csvs()
        st.rerun()

# Добавляем информацию о загруженных данных в сайдбар
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Статистика данных")
if not products.empty:
    st.sidebar.write(f"Продукты: {len(products)}")
if not samples.empty:
    st.sidebar.write(f"Партии: {len(samples)}")
if not measurements.empty:
    st.sidebar.write(f"Измерения: {len(measurements)}")

st.sidebar.markdown("---")
st.sidebar.info("""
**Использование:**
1. Выберите продукт на главной странице
2. Изучите процесс производства
3. Анализируйте данные в разделе аналитики
4. Добавляйте новые данные через формы
""")

# Кнопка для сброса состояния
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Сбросить состояние приложения"):
    st.session_state.clear()
    st.rerun()
