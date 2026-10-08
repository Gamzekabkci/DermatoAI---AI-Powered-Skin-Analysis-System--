import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def verileri_hazirla(resim_yolu, maske_yolu, limit=500):
    resim_listesi, maske_listesi = [], []
    # Dosyaları isim sırasına göre diziyoruz ki resim ve maske eşleşsin
    dosyalar = sorted([f for f in os.listdir(resim_yolu) if f.endswith('.jpg')])[:limit]

    for dosya_adi in dosyalar:
        # Resim Yükleme ve Normalizasyon
        img = cv2.imread(os.path.join(resim_yolu, dosya_adi))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128)) / 255.0

        # Maske Yükleme (Siyah-Beyaz)
        mask_adi = dosya_adi.replace(".jpg", "_segmentation.png")
        mask = cv2.imread(os.path.join(maske_yolu, mask_adi), cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)  # (128, 128, 1) kanal ekleme

            resim_listesi.append(img)
            maske_listesi.append(mask)

    return np.array(resim_listesi), np.array(maske_listesi)

# Kullanım:
# X, y = verileri_yukle(RESIM_YOLU, MASKE_YOLU)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)