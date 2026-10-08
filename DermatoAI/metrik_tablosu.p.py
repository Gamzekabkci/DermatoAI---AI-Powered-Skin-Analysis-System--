import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from veri_hazirlik import verileri_hazirla
from sklearn.model_selection import train_test_split

# --- 1. VERİLERİ VE MODELLERİ YÜKLE ---
RESIM_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1-2_Training_Input'
MASKE_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1_Training_GroundTruth'

print("Veriler yükleniyor...")
X, y = verileri_hazirla(RESIM_DIR, MASKE_DIR, limit=None)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# YENİ MODELLERİNİ BURADAN YÜKLÜYORUZ
print("Modeller hafızaya alınıyor...")
model_unet = tf.keras.models.load_model('deri_analiz_final.h5')
model_trans = tf.keras.models.load_model('deri_analiz_transformer.h5') # Yeni modelin

def metrik_hesapla(model, X_test, y_test):
    print(f"{model.name} için tahminler üretiliyor...")
    tahminler = model.predict(X_test, verbose=0)
    dice_list = []
    jaccard_list = []

    for i in range(len(X_test)):
        y_true = y_test[i].flatten()
        y_pred = (tahminler[i].flatten() > 0.5).astype(np.float32)

        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)

        # Matematiksel olarak doğru Dice ve Jaccard hesaplama
        dice = (2. * intersection) / (union + 1e-7)
        jaccard = intersection / (np.sum((y_true + y_pred) > 0) + 1e-7)

        dice_list.append(dice)
        jaccard_list.append(jaccard)

    return np.mean(dice_list), np.mean(jaccard_list)

# --- 2. ANALİZ ---
u_dice, u_jacc = metrik_hesapla(model_unet, X_test, y_test)
t_dice, t_jacc = metrik_hesapla(model_trans, X_test, y_test)

# Tablo Verisi
data = {
    'Performans Metriği': ['Ortalama Dice Skoru (F1)', 'Ortalama Jaccard (IoU)'],
    'Klasik U-Net': [f"{u_dice:.4f}", f"{u_jacc:.4f}"],
    'Transformer-Attention U-Net': [f"{t_dice:.4f}", f"{t_jacc:.4f}"]
}

df = pd.DataFrame(data)

# --- 3. PROFESYONEL TABLO OLUŞTURMA ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')


colors = [["#ffffff", "#f8f9fa", "#e3f2fd"], ["#ffffff", "#f8f9fa", "#e3f2fd"]]

tablo = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 colColours=["#2c3e50", "#2c3e50", "#2c3e50"])


for (row, col), cell in tablo.get_celld().items():
    if row == 0:
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')

tablo.auto_set_font_size(False)
tablo.set_fontsize(11)
tablo.scale(1.2, 2.5)

plt.title("Modellerin Test Seti Üzerindeki Segmentasyon Başarısı", pad=30, fontweight='bold', fontsize=14)
plt.savefig("model_karsilastirma_tablosu.png", dpi=300, bbox_inches='tight')
print("\nAnaliz Tamamlandı!")
print(f"U-Net Dice: {u_dice:.4f}")
print(f"Transformer Dice: {t_dice:.4f}")
print("Tablo 'model_karsilastirma_tablosu.png' olarak kaydedildi!")
plt.show()