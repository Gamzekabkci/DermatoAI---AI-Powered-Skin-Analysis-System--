import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from veri_hazirlik import verileri_hazirla
from sklearn.model_selection import train_test_split

# --- 1. VERİLERİ VE MODELİ YÜKLE ---
# Yeni model ismine göre güncelledik
MODEL_ISMI = 'deri_analiz_final.h5'
RESIM_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1-2_Training_Input'
MASKE_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1_Training_GroundTruth'

print("Test verileri hazırlanıyor...")
# Tüm veri setini çekip içinden test ayırıyoruz
X, y = verileri_hazirla(RESIM_DIR, MASKE_DIR, limit=None)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"{MODEL_ISMI} yükleniyor...")
model = tf.keras.models.load_model(MODEL_ISMI)


# --- 2. ANALİZ FONKSİYONU ---
def final_basari_analizi(model, X_test, y_test):
    print("Tahminler oluşturuluyor (n=" + str(len(X_test)) + ")...")
    tahminler = model.predict(X_test)

    dice_skorlari = []
    jaccard_skorlari = []

    for i in range(len(X_test)):
        y_true = y_test[i]
        y_pred = (tahminler[i] > 0.5).astype(np.float32)

        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)

        # Dice Katsayısı
        dice = (2. * intersection) / (union + 1e-7)
        dice_skorlari.append(dice)

        # Jaccard (IoU) İndeksi
        jaccard = intersection / (union - intersection + 1e-7)
        jaccard_skorlari.append(jaccard)

    print("\n" + "=" * 45)
    print("      📊 FINAL MODEL PERFORMANS RAPORU      ")
    print("-" * 45)
    print(f"Ortalama DICE Skoru:    {np.mean(dice_skorlari):.4f}")
    print(f"Ortalama JACCARD (IoU): {np.mean(jaccard_skorlari):.4f}")
    print("=" * 45)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(y_test[0].squeeze(), cmap='gray')
    plt.title("Hocanın KG'si (Gerçek)")

    plt.subplot(1, 2, 2)
    plt.imshow(tahminler[0].squeeze(), cmap='gray')
    plt.title("Hocanın T'si (Tahmin)")
    plt.show()

    #dice IoU
    
# Analizi Başlat
final_basari_analizi(model, X_test, y_test)