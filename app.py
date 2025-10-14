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
# --- MODELS & ANALYTICS ---
# ---------------------------
elif st.session_state['page'] == 'Модели и аналитика':
    st.title("📊 Модели и аналитика")
    st.write("Построение регрессионных моделей и визуализация результатов по партиям/измерениям.")

    if measurements.empty or samples.empty:
        st.warning("Measurements.csv и/или Samples.csv пусты — загрузите данные.")
    else:
        meas = measurements.copy()
        if 'actual_numeric' not in meas.columns and 'actual_value' in meas.columns:
            meas['actual_numeric'] = meas['actual_value'].apply(parse_numeric)
        pivot = meas.pivot_table(index='sample_id', columns='parameter', values='actual_numeric', aggfunc='first').reset_index()
        df_all = samples.merge(pivot, on='sample_id', how='left')

        st.subheader("Данные (preview)")
        st.dataframe(df_all.head(50))

        # choose product filter optionally
        prod_options = ["Все"]
        if not products.empty and 'product_id' in products.columns and 'name' in products.columns:
            for _, r in products[['product_id','name']].dropna().iterrows():
                try:
                    prod_options.append(f"{int(r['product_id'])} - {r['name']}")
                except Exception:
                    pass
        prod_filter = st.selectbox("Фильтр по продукту (опционально)", options=prod_options, index=0)
        if prod_filter != "Все":
            try:
                pidf = int(prod_filter.split(' - ')[0])
                df_all = df_all[df_all['product_id'] == pidf]
            except Exception:
                pass

        ignore_cols = ['sample_id','product_id','reg_number','date_received','storage_days','conditions','notes']
        possible = [c for c in df_all.columns if c not in ignore_cols]
        if not possible:
            st.warning("Нет признаков для моделирования (посмотри Measurements.csv и Samples.csv).")
        else:
            target = st.selectbox("Target (целевая переменная)", options=possible, index=0)
            features_default = [c for c in ['Белок','Жир','Влага','storage_days'] if c in df_all.columns][:3]
            features = st.multiselect("Features (признаки)", options=[c for c in possible if c != target], default=features_default)

            st.markdown("**Параметры обучения**")
            test_size = st.slider("Тестовая доля", 0.1, 0.5, 0.3)
            scale_display = st.checkbox("Показать корреляционную матрицу", value=True)

            if not features:
                st.info("Выберите как минимум один признак.")
            else:
                dataset = df_all[[target] + features].dropna()
                st.write("Строк доступных для обучения:", len(dataset))
                if len(dataset) < 5:
                    st.warning("Недостаточно данных (нужно ≥5 строк).")
                else:
                    X = dataset[features].astype(float).values
                    y = dataset[target].astype(float).values

                    if scale_display:
                        st.subheader("Корреляционная матрица (features + target)")
                        corr = dataset.corr(numeric_only=True)
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="RdYlBu_r", center=0)
                        st.pyplot(fig)

                    if SKLEARN:
                        st.subheader("Настройки модели")
                        col1, col2 = st.columns(2)
                        with col1:
                            alg = st.selectbox("Алгоритм", ["Linear","Ridge","Lasso"], index=0)
                        with col2:
                            random_state = st.number_input("Random State", value=42, min_value=0)

                        alpha = None
                        if alg in ["Ridge","Lasso"]:
                            alpha = st.number_input("alpha (регуляризация)", value=1.0, format="%.4f", min_value=0.0)

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                        Model = LinearRegression if alg=="Linear" else (Ridge if alg=="Ridge" else Lasso)
                        model = Model() if alpha is None else Model(alpha=alpha)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)

                        st.subheader("📊 Результаты модели")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² (точность)", f"{r2:.4f}",
                                      delta="Хорошо" if r2 > 0.7 else "Удовл." if r2 > 0.5 else "Слабо")
                        with col2:
                            st.metric("RMSE (ошибка)", f"{rmse:.4f}")
                        with col3:
                            st.metric("MAE (ошибка)", f"{mae:.4f}")

                        # Доп. метрика MAPE (если возможно)
                        if np.all(np.abs(y_test) > 1e-12):
                            mape = np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-12, None))) * 100
                            st.metric("MAPE (%)", f"{mape:.2f}")

                        # coefficients
                        try:
                            coefs = dict(zip(features, np.atleast_1d(model.coef_)))
                            intercept_val = float(model.intercept_)

                            st.subheader("🔍 Коэффициенты модели")
                            coef_df = pd.DataFrame.from_dict(coefs, orient='index', columns=['Коэффициент'])
                            coef_df['Влияние'] = coef_df['Коэффициент'].apply(
                                lambda x: '📈 Положительное' if x > 0 else '📉 Отрицательное' if x < 0 else '➖ Нулевое'
                            )
                            st.dataframe(coef_df.round(6))

                            # Equation
                            equation = f"{target} = "
                            for i, (feat, coef) in enumerate(coefs.items()):
                                if i == 0:
                                    equation += f"{coef:.4f}×{feat}"
                                else:
                                    sign = " + " if coef >= 0 else " - "
                                    equation += f"{sign}{abs(coef):.4f}×{feat}"
                            equation += f" + {intercept_val:.4f}"
                            st.code(equation, language='python')

                        except Exception as e:
                            st.warning(f"Не удалось получить коэффициенты: {e}")

                        # Visualizations
                        st.subheader("📈 Визуализации")

                        # Actual vs Predicted and Residuals
                        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                        ax1.scatter(y_test, y_pred, alpha=0.7)
                        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                        ax1.plot(lims, lims, 'r--', lw=2)
                        ax1.set_xlabel("Фактические значения")
                        ax1.set_ylabel("Предсказанные значения")
                        ax1.set_title("Фактические vs Предсказанные")
                        ax1.grid(True, alpha=0.3)

                        residuals = y_test - y_pred
                        ax2.scatter(y_pred, residuals, alpha=0.7)
                        ax2.axhline(y=0, color='red', linestyle='--')
                        ax2.set_xlabel("Предсказанные значения")
                        ax2.set_ylabel("Остатки")
                        ax2.set_title("Остатки модели")
                        ax2.grid(True, alpha=0.3)

                        st.pyplot(fig1)

                        # Feature importance if multiple features
                        if len(features) > 1:
                            st.subheader("📊 Важность признаков")
                            imp_vals = np.abs(np.atleast_1d(model.coef_))
                            importance_df = pd.DataFrame({
                                'Признак': features,
                                'Важность': imp_vals
                            }).sort_values('Важность', ascending=False)
                            if importance_df['Важность'].max() > 0:
                                importance_df["Отн. важность"] = importance_df["Важность"] / importance_df["Важность"].max()

                            fig2, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(importance_df['Признак'], importance_df['Важность'])
                            ax.set_xlabel('Важность (|коэффициента|)')
                            ax.set_title('Относительная важность признаков')

                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                        f'{width:.4f}', ha='left', va='center')

                            st.pyplot(fig2)

                        # Single feature regression plot
                        if len(features) == 1:
                            st.subheader("📈 График зависимости")
                            fig3, ax = plt.subplots(figsize=(10, 6))

                            x_all = dataset[features[0]].astype(float).values
                            y_all = dataset[target].astype(float).values

                            ax.scatter(x_all, y_all, alpha=0.6, label='Данные')

                            xs = np.linspace(x_all.min(), x_all.max(), 100)
                            coef_val = float(np.atleast_1d(model.coef_)[0])
                            intercept_val = float(model.intercept_)
                            ax.plot(xs, coef_val*xs + intercept_val, color='red', linewidth=2, label='Линия регрессии')

                            ax.set_xlabel(features[0])
                            ax.set_ylabel(target)
                            ax.set_title(f'Зависимость {target} от {features[0]}')
                            ax.legend()
                            ax.grid(True, alpha=0.3)

                            st.pyplot(fig3)

                            # Prediction interface
                            st.subheader("🔮 Прогнозирование")
                            col1, col2 = st.columns(2)
                            with col1:
                                input_val = st.number_input(
                                    f"Введите значение {features[0]}",
                                    value=float(np.nanmean(x_all)) if len(x_all) else 0.0,
                                    format="%.4f"
                                )
                            with col2:
                                prediction = coef_val * input_val + intercept_val
                                st.metric(f"Предсказанное значение {target}", f"{prediction:.4f}")

                    else:
                        # fallback: only single feature regression with numpy.polyfit
                        if len(features) == 1:
                            st.warning("⚠️ scikit-learn не установлен — используется простая линейная регрессия numpy")
                            x = X.flatten()
                            coef = np.polyfit(x, y, 1)
                            slope, intercept = float(coef[0]), float(coef[1])

                            st.success("**Уравнение регрессии:**")
                            st.code(f"{target} = {slope:.4f} × {features[0]} + {intercept:.4f}", language='python')

                            y_pred_simple = slope * x + intercept
                            r2_simple = 1 - np.sum((y - y_pred_simple)**2) / np.sum((y - np.mean(y))**2)
                            rmse_simple = np.sqrt(np.mean((y - y_pred_simple)**2))

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("R²", f"{r2_simple:.4f}")
                            with col2:
                                st.metric("RMSE", f"{rmse_simple:.4f}")

                            fig, ax = plt.subplots(figsize=(10, 6))
                            idx = np.argsort(x)
                            ax.scatter(x, y, alpha=0.7, label='Данные')
                            ax.plot(x[idx], np.polyval(coef, x[idx]), color='red', linewidth=2, label='Регрессия')
                            ax.set_xlabel(features[0])
                            ax.set_ylabel(target)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)

                            # Prediction interface
                            st.subheader("🔮 Прогнозирование")
                            input_val = st.number_input(
                                f"Введите значение {features[0]}",
                                value=float(np.nanmean(x)) if len(x) else 0.0,
                                format="%.4f"
                            )
                            prediction = slope * input_val + intercept
                            st.metric(f"Предсказанное значение {target}", f"{prediction:.4f}")

                        else:
                            st.error("""
                            ❌ Для работы с несколькими признаками требуется scikit-learn

                            Установите: `pip install scikit-learn`

                            Или используйте только один признак для анализа.
                            """)

    st.markdown("---")
    st.subheader("💡 Рекомендации по улучшению моделей")

    tips = [
        "✅ Добавьте больше партий и измерений в CSV файлы",
        "✅ Убедитесь, что данные содержат минимальные выбросы",
        "✅ Проверьте корреляцию между признаками перед моделированием",
        "✅ Используйте перекрестную проверку для оценки устойчивости модели",
        "✅ Рассмотрите возможность добавления полиномиальных признаков",
        "✅ Нормализуйте данные для лучшей сходимости алгоритмов"
    ]
    for tip in tips:
        st.write(tip)

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
