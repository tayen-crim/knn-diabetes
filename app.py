import math
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prediksi Risiko Diabetes dengan KNN",
    page_icon="🩺",
    layout="wide",
)

# ------------------------------------------------------------------
# Tampilan: tema terang ala lembar hasil laboratorium
# Palet: kertas #F4F6F9, tinta biru #1F3556, garis #D5DCE6,
#        merah klinis #B93A2B (risiko tinggi), hijau daun #2E7A58 (risiko rendah)
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;500;600&display=swap');

    :root {
        --paper: #F4F6F9;
        --card: #FFFFFF;
        --ink: #1F3556;
        --muted: #5F6B7D;
        --line: #D5DCE6;
        --high: #B93A2B;
        --high-bg: #FBEDEA;
        --low: #2E7A58;
        --low-bg: #E9F4EE;
    }

    html, body, .stApp, [data-testid="stMarkdownContainer"], label, p, li {
        font-family: 'Source Sans 3', system-ui, sans-serif;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.4rem; max-width: 1180px; }

    h1, h2, h3 {
        font-family: 'Bricolage Grotesque', 'Source Sans 3', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
    }
    h1 { font-weight: 700 !important; font-size: 2.3rem !important; margin-bottom: 0.2rem; }
    h2 { font-weight: 700 !important; font-size: 1.45rem !important; }
    h3 { font-weight: 500 !important; font-size: 1.15rem !important; }

    .lead { color: var(--muted); font-size: 1.05rem; max-width: 62ch; margin-bottom: 1.6rem; }

    /* Kartu */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        box-shadow: none;
    }

    /* Input */
    [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
        background: #FFFFFF !important;
        border-color: var(--line) !important;
        border-radius: 6px !important;
    }
    label p { color: var(--ink) !important; font-weight: 600; }

    /* Tombol */
    div.stButton > button {
        width: 100%;
        background: var(--ink); color: #FFFFFF;
        border: 1px solid var(--ink); border-radius: 6px;
        padding: 0.65rem 1.1rem; font-weight: 600; font-size: 1rem;
    }
    div.stButton > button:hover { background: #2B4A78; border-color: #2B4A78; color: #FFFFFF; }
    div.stButton > button:focus-visible { outline: 3px solid #8FB0E3; outline-offset: 2px; }

    /* Hasil */
    .verdict {
        border-left: 6px solid; border-radius: 6px;
        padding: 18px 22px; margin: 4px 0 18px 0;
    }
    .verdict .kelas { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.7rem; line-height: 1.15; }
    .verdict .ket { margin-top: 4px; font-size: 1rem; }
    .verdict.high { background: var(--high-bg); border-color: var(--high); color: var(--high); }
    .verdict.low  { background: var(--low-bg);  border-color: var(--low);  color: var(--low); }

    /* Barisan tetangga: inti tampilan hasil */
    .neighbors { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 4px 0; }
    .chip {
        display: flex; align-items: center; gap: 10px;
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px;
        padding: 6px 14px 6px 8px;
    }
    .chip .dot {
        width: 28px; height: 28px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        color: #FFFFFF; font-size: 0.78rem; font-weight: 600;
    }
    .chip .dot.k0 { background: var(--low); }
    .chip .dot.k1 { background: var(--high); }
    .chip .meta { font-size: 0.9rem; color: var(--muted); }
    .chip .meta b { color: var(--ink); font-weight: 600; }

    .legend { color: var(--muted); font-size: 0.9rem; margin-bottom: 14px; }
    .legend span.k0, .legend span.k1 { font-weight: 600; }
    .legend span.k0 { color: var(--low); }
    .legend span.k1 { color: var(--high); }

    .empty {
        color: var(--muted); padding: 40px 8px; text-align: left; font-size: 1.02rem;
    }
    .note { color: var(--muted); font-size: 0.88rem; }

    [data-testid="stExpander"] { background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("Prediksi Risiko Diabetes dengan KNN")
st.markdown(
    '<p class="lead">Masukkan data seorang pasien, lalu lihat pasien latih mana yang paling mirip '
    'dan kelas risiko yang dipilih mayoritas tetangganya.</p>',
    unsafe_allow_html=True,
)

# Data pelatihan sesuai studi kasus pada PDF
data = [
    {"Pasien": "P01", "Usia": 25, "BMI": 21.5, "Glukosa": 90,  "Tekanan Darah": 110, "Riwayat Keluarga": 0, "Kelas": 0},
    {"Pasien": "P02", "Usia": 30, "BMI": 23.1, "Glukosa": 95,  "Tekanan Darah": 115, "Riwayat Keluarga": 0, "Kelas": 0},
    {"Pasien": "P03", "Usia": 35, "BMI": 25.0, "Glukosa": 105, "Tekanan Darah": 120, "Riwayat Keluarga": 0, "Kelas": 0},
    {"Pasien": "P04", "Usia": 40, "BMI": 27.5, "Glukosa": 120, "Tekanan Darah": 125, "Riwayat Keluarga": 1, "Kelas": 1},
    {"Pasien": "P05", "Usia": 45, "BMI": 30.2, "Glukosa": 140, "Tekanan Darah": 135, "Riwayat Keluarga": 1, "Kelas": 1},
    {"Pasien": "P06", "Usia": 50, "BMI": 31.5, "Glukosa": 155, "Tekanan Darah": 140, "Riwayat Keluarga": 1, "Kelas": 1},
    {"Pasien": "P07", "Usia": 55, "BMI": 33.0, "Glukosa": 165, "Tekanan Darah": 145, "Riwayat Keluarga": 1, "Kelas": 1},
    {"Pasien": "P08", "Usia": 28, "BMI": 22.5, "Glukosa": 92,  "Tekanan Darah": 112, "Riwayat Keluarga": 0, "Kelas": 0},
    {"Pasien": "P09", "Usia": 38, "BMI": 26.8, "Glukosa": 115, "Tekanan Darah": 122, "Riwayat Keluarga": 0, "Kelas": 0},
    {"Pasien": "P10", "Usia": 48, "BMI": 29.5, "Glukosa": 145, "Tekanan Darah": 130, "Riwayat Keluarga": 1, "Kelas": 1},
]
df = pd.DataFrame(data)
fitur = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

# ------------------------------------------------------------------
# Bagian utama: formulir di kiri, hasil di kanan
# ------------------------------------------------------------------
col_form, col_result = st.columns([5, 7], gap="large")

with col_form:
    with st.container(border=True):
        st.header("Data pasien baru")
        c1, c2 = st.columns(2)
        with c1:
            usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
            glukosa = st.number_input("Glukosa", min_value=1, max_value=1000, value=130, step=1)
            riwayat_label = st.selectbox(
                "Riwayat keluarga", options=[0, 1], index=1,
                format_func=lambda v: "1 - Ada" if v == 1 else "0 - Tidak ada",
            )
        with c2:
            bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=28.5, step=0.1, format="%.2f")
            tekanan = st.number_input("Tekanan darah", min_value=1, max_value=350, value=128, step=1)
            k = st.selectbox("Jumlah tetangga (K)", options=[3, 5, 7], index=0)

        hitung = st.button(f"Hitung KNN (K={k})", type="primary")

with col_result:
    with st.container(border=True):
        st.header("Hasil prediksi")

        if not hitung:
            st.markdown(
                '<div class="empty">Belum ada hasil. Isi data pasien di sebelah kiri, '
                'lalu tekan <b>Hitung KNN</b>.</div>',
                unsafe_allow_html=True,
            )
        else:
            pasien_baru = {
                "Usia": float(usia),
                "BMI": float(bmi),
                "Glukosa": float(glukosa),
                "Tekanan Darah": float(tekanan),
                "Riwayat Keluarga": float(riwayat_label),
            }
            distances = []

            for _, row in df.iterrows():
                distance = math.sqrt(sum(
                    (pasien_baru[col] - float(row[col])) ** 2 for col in fitur
                ))
                distances.append({
                    "Pasien": row["Pasien"],
                    "Usia": row["Usia"],
                    "BMI": row["BMI"],
                    "Glukosa": row["Glukosa"],
                    "Tekanan Darah": row["Tekanan Darah"],
                    "Riwayat Keluarga": row["Riwayat Keluarga"],
                    "Kelas": int(row["Kelas"]),
                    "Jarak Euclidean": distance,
                })

            distances = sorted(distances, key=lambda x: x["Jarak Euclidean"])
            top_k = distances[:k]
            count_0 = sum(1 for item in top_k if item["Kelas"] == 0)
            count_1 = sum(1 for item in top_k if item["Kelas"] == 1)
            if count_0 == count_1:
                prediction = top_k[0]["Kelas"]
            else:
                prediction = 0 if count_0 > count_1 else 1

            if prediction == 1:
                st.markdown(
                    f'<div class="verdict high"><div class="kelas">Kelas 1: risiko tinggi</div>'
                    f'<div class="ket">{count_1} dari {k} tetangga terdekat berkelas tinggi.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict low"><div class="kelas">Kelas 0: risiko rendah</div>'
                    f'<div class="ket">{count_0} dari {k} tetangga terdekat berkelas rendah.</div></div>',
                    unsafe_allow_html=True,
                )

            st.subheader(f"{k} tetangga terdekat")
            chips = "".join(
                f'<div class="chip"><span class="dot k{item["Kelas"]}">{item["Pasien"][1:]}</span>'
                f'<span class="meta"><b>{item["Pasien"]}</b> &nbsp;jarak {item["Jarak Euclidean"]:.2f}</span></div>'
                for item in top_k
            )
            st.markdown(f'<div class="neighbors">{chips}</div>', unsafe_allow_html=True)
           
            nearest_df = pd.DataFrame(top_k)
            nearest_df["Kelas"] = nearest_df["Kelas"].map({0: "0 - Rendah", 1: "1 - Tinggi"})
            nearest_df["Jarak Euclidean"] = nearest_df["Jarak Euclidean"].map(lambda x: f"{x:.4f}")
            st.dataframe(nearest_df, use_container_width=True, hide_index=True)

            st.write(f"**Hasil voting:** kelas 0 (rendah) = {count_0} suara, kelas 1 (tinggi) = {count_1} suara.")

# ------------------------------------------------------------------
# Data latih & penjelasan
# ------------------------------------------------------------------
st.write("")
with st.container(border=True):
    st.header("Data latih pasien")
    display_df = df.copy()
    display_df["Kelas"] = display_df["Kelas"].map({0: "0 - Rendah", 1: "1 - Tinggi"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)
