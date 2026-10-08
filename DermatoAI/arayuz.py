import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from groq import Groq

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DermatoAI - Deri Analiz Sistemi", layout="wide")


st.markdown("""
    <style>
    /* Arka Plan */
    .stApp {
        background-color: #f4f7f9;
    }

    /* Resim Kartları: Kesin Boyut ve Modern Gölgelendirme */
    [data-testid="stImage"] img {
        width: 350px !important;
        height: 350px !important;
        object-fit: fill !important;
        border-radius: 24px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        border: 5px solid white;
    }

    /* Modern Dice ve Referans Kutuları */
    .dice-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; padding: 12px; border-radius: 12px;
        text-align: center; font-weight: 700; font-size: 18px;
        margin-top: 10px; box-shadow: 0 4px 10px rgba(30, 60, 114, 0.2);
    }

    .referans-container {
        background: linear-gradient(135deg, #0f9b0f 0%, #00cc00 100%);
        color: white; padding: 12px; border-radius: 12px;
        text-align: center; font-weight: 700; font-size: 18px;
        margin-top: 10px;
    }

    /* TÜRKÇE VE GÜZEL GÖRÜNÜMLÜ ÜST BAŞLIK */
    .hero-section {
        background: linear-gradient(to right, #2c3e50, #3498db);
        padding: 40px; border-radius: 20px; text-align: center;
        margin-bottom: 30px; box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }

    .hero-title {
        color: white; font-family: 'Helvetica Neue', sans-serif;
        font-weight: 900; font-size: 50px; margin: 0; letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #d1d9e6; font-size: 18px; font-weight: 300; margin-top: 10px;
    }

    /* YENİ MODERN RAPOR KARTI TASARIMI */
    .modern-report {
        background: #ffffff;
        border-radius: 20px;
        padding: 35px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 20px 40px rgba(30, 60, 114, 0.08);
        font-family: 'Segoe UI', Tahoma, sans-serif;
        color: #2c3e50;
        position: relative;
        overflow: hidden;
    }

    .modern-report::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 6px; height: 100%;
        background: linear-gradient(180deg, #3498db, #1e3c72);
    }

    .report-badge {
        position: absolute;
        top: 25px; right: 25px;
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white; padding: 6px 16px; border-radius: 30px;
        font-size: 12px; font-weight: 800; letter-spacing: 1px;
        box-shadow: 0 4px 10px rgba(0, 180, 219, 0.3);
    }

    .report-title-box {
        display: flex; align-items: center; margin-bottom: 25px;
        border-bottom: 2px solid #f0f4f8; padding-bottom: 15px;
    }

    .report-title-box h2 {
        margin: 0; color: #1e3c72; font-size: 24px;
        font-weight: 900; margin-left: 15px; letter-spacing: -0.5px;
    }

    .report-text {
        font-size: 15px; line-height: 1.8; color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# ÜST PANEL
st.markdown("""
    <div class='hero-section'>
        <h1 class='hero-title'>DermatoAI: Akıllı Analiz</h1>
        <p class='hero-subtitle'>Yapay Zeka Destekli Deri Lezyonu Segmentasyon ve Klinik Karar Destek Paneli</p>
    </div>
    """, unsafe_allow_html=True)


# --- CANLI LLM RAPOR ÜRETME FONKSİYONU ---
def llm_rapor_olustur(score_u, score_t):
    try:

        client = Groq(api_key="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")

        prompt = f"""
        Sen kıdemli bir dermatologsun. Önündeki bilgisayar destekli tanı sistemi, şüpheli bir deri lezyonunu (melanom/malignite şüphesi) inceledi ve iki farklı klinik haritalama (segmentasyon) sonucu üretti.
        Sistemlerin uzman klinisyen çizimiyle olan anatomik marj uyumu (Dice Skorları) şu şekildedir:
        - 1. Sistem Uyum Doğruluğu: {score_u:.4f}
        - 2. Sistem Uyum Doğruluğu: {score_t:.4f}

        Bu verilere ve dermoskopik muayene prensiplerine dayanarak, hastanın takip dosyasına eklenecek resmi ve Türkçe bir klinik rapor yaz.

        RAPOR FORMATI VE KURALLARI:
        1. KESİNLİKLE yazılımsal, algoritmik veya bilgisayar mühendisliği terimleri (U-Net, Transformer, CNN, kod, piksel vb.) KULLANMA.
        2. Raporu şu tıbbi alt başlıklarla yapılandır:
           - ### 🔍 Makroskobik ve Dermoskopik Bulgular: Lezyonun asimetrisi ve çevre dokuyla olan kontrastı hakkında klinik yorum yap. 
           - ### 🎯 Artefakt ve Parazit Değerlendirmesi: Deri üzerindeki tüylerin sınır tespitini nasıl zorlaştırdığını ve sistemlerin bu parazitleri nasıl yönettiğini tıbbi dille anlat.
           - ### 🩺 Sınır Güvenilirliği ve Cerrahi Planlama: Hangi sistemin belirlediği sınırın eksizyon marjları için güvenli olduğunu skorlara dayanarak açıkla.
           - ### 📋 Klinik Karar ve Öneri: Hastanın biyopsi veya takip sürecine yönelik nihai doktor tavsiyeni yaz.
        3. Metin doğrudan '### Makroskobik ve Dermoskopik Bulgular' ile başlasın, gereksiz selamlaşma veya giriş cümlesi kurma.
        """

        tamamlama = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        return tamamlama.choices[0].message.content
    except Exception as e:
        return f"### ❌ LLM Bağlantı Hatası\nAPI anahtarı eksik veya hatalı olabilir. Lütfen kodu kontrol edin.\n\n*Hata Detayı: {e}*"


# --- MODELLERİ YÜKLEME ---
@st.cache_resource
def modelleri_yukle():
    m1 = tf.keras.models.load_model('deri_analiz_final.h5')
    m2 = tf.keras.models.load_model('deri_analiz_transformer.h5')
    return m1, m2


unet_model, trans_model = modelleri_yukle()

# --- DOSYA YÜKLEME ---
c1, c2 = st.columns(2)
with c1:
    orijinal_dosya = st.file_uploader("🖼️ Analiz Edilecek Görüntü", type=["jpg", "jpeg", "png"])
with c2:
    maske_dosya = st.file_uploader("🧪 Uzman Maskesi (Gerçek Değer)", type=["jpg", "jpeg", "png"])


def dice_hesapla(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-7)


# --- ANALİZ VE GÖRSELLEŞTİRME ---
if orijinal_dosya:
    img = Image.open(orijinal_dosya).convert('RGB')
    img_input = np.expand_dims(cv2.resize(np.array(img), (128, 128)) / 255.0, axis=0)

    m_u = (unet_model.predict(img_input, verbose=0)[0] > 0.5).astype(np.float32)
    m_t = (trans_model.predict(img_input, verbose=0)[0] > 0.5).astype(np.float32)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4 if maske_dosya else 3)

    with cols[0]:
        st.markdown("### 📸 Orijinal")
        st.image(img)

    gt_maske = None
    if maske_dosya:
        gt_img = Image.open(maske_dosya).convert('L')
        gt_maske = (cv2.resize(np.array(gt_img), (128, 128)) / 255.0 > 0.5).astype(np.float32)

    with cols[1]:
        st.markdown("### 🧬 Klasik U-Net")
        st.image(m_u)
        if gt_maske is not None:
            score_u = dice_hesapla(gt_maske, m_u)
            st.markdown(f"<div class='dice-container'>Dice Skoru: {score_u:.4f}</div>", unsafe_allow_html=True)

    with cols[2]:
        st.markdown("### 🤖 Transformer")
        st.image(m_t)
        if gt_maske is not None:
            score_t = dice_hesapla(gt_maske, m_t)
            st.markdown(f"<div class='dice-container'>Dice Skoru: {score_t:.4f}</div>", unsafe_allow_html=True)

    if maske_dosya:
        with cols[3]:
            st.markdown("### 🎯 Uzman Maskesi")
            st.image(gt_maske)
            st.markdown("<div class='referans-container'>Referans Bilgi</div>", unsafe_allow_html=True)

    # --- GERÇEK ZAMANLI AKILLI RAPOR PANELİ ---
    if maske_dosya:
        st.markdown("---")
        if st.button("📄 Yapay Zeka ile Klinik Rapor Taslağı Oluştur"):
            with st.spinner("Llama 3.1 verileri klinik düzeyde analiz ediyor, lütfen bekleyin..."):
                klinik_rapor = llm_rapor_olustur(score_u, score_t)

                # LLM'in markdown başlıklarını şık HTML'e çeviriyoruz
                duzenli_rapor = klinik_rapor.replace('###', '<br><b style="color:#2980b9; font-size:18px;">').replace(
                    '\n', '<br>')

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='modern-report'>
                    <div class='report-badge'>🤖 LLAMA 3.1 YZ ASİSTAN</div>
                    <div class='report-title-box'>
                        <span style='font-size: 32px;'>🩺</span>
                        <h2>Dermatolojik Klinik Değerlendirme Raporu</h2>
                    </div>
                    <div class='report-text'>
                        {duzenli_rapor}
                    </div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("Lütfen bir resim yükleyerek analizi başlatın.")