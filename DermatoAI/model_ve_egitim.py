import tensorflow as tf
from veri_hazirlik import verileri_hazirla
from sklearn.model_selection import train_test_split
import os

# --- 1. TÜM VERİLERİ YÜKLE ---
RESIM_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1-2_Training_Input'
MASKE_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1_Training_GroundTruth'

# Limit=None ile klasördeki 2594 resmin tamamı okunur
print("Veriler yükleniyor, lütfen bekleyin...")
X, y = verileri_hazirla(RESIM_DIR, MASKE_DIR, limit=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 2. U-NET MİMARİSİ  ---
def unet_olustur(input_size=(128, 128, 3)):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    # Bridge
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)

    # Decoder
    u3 = tf.keras.layers.UpSampling2D((2, 2))(c2)
    c3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u3)

    # Çıkış Katmanı (Sigmoid ile pikselleri 0-1 arasına sıkıştırıyoruz)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- 3. EĞİTİM STRATEJİSİ (Early Stopping) ---
model = unet_olustur()

# Model artık gelişmiyorsa (val_loss 5 epoch boyunca düşmezse) eğitimi durdurur
# restore_best_weights=True: En iyi performansı veren ağırlıkları geri yükler
erken_durdurma = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\nEğitim başlıyor... (Tüm veri seti + 50 Max Epoch)")
# 50 Epoch hedefliyoruz ama erken durdurma sayesinde model doyunca kendi bitecek
tarihce = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=16,
    callbacks=[erken_durdurma],
    verbose=1
)

# --- 4. MODELİ KAYDET ---
model_ismi = 'deri_analiz_final.h5'
model.save(model_ismi)
print(f"\nİşlem Tamamlandı! En iyi model '{model_ismi}' adıyla kaydedildi.")