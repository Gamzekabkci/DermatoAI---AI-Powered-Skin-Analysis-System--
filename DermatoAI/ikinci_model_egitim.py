import tensorflow as tf
from veri_hazirlik import verileri_hazirla
from sklearn.model_selection import train_test_split

# --- 1. VERİLERİ YÜKLE ---
# Masaüstündeki veri seti klasör yolların
RESIM_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1-2_Training_Input'
MASKE_DIR = r'C:\Users\gkaba\Desktop\veri_seti\ISIC2018_Task1_Training_GroundTruth'

print("Veriler yükleniyor, lütfen bekleyin...")
X, y = verileri_hazirla(RESIM_DIR, MASKE_DIR, limit=None)  # Tüm veri seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 2. TRANSFORMER / ATTENTION MEKANİZMASI ---
# Bu kısım modelin 'nereye bakması gerektiğini' öğrenmesini sağlar
import tensorflow as tf


def attention_gate(x, gating, inter_shape):
    # theta_x: x'i (aşağıdan gelen) gating boyutuna çekmek için stride kullanıyoruz
    # Eğer x ve gating boyutları farklıysa stride ile eşitliyoruz
    stride_h = x.shape[1] // gating.shape[1]
    stride_w = x.shape[2] // gating.shape[2]

    theta_x = tf.keras.layers.Conv2D(inter_shape, (stride_h, stride_w), strides=(stride_h, stride_w), padding='same')(x)
    phi_g = tf.keras.layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)

    concat_xg = tf.keras.layers.add([theta_x, phi_g])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)
    psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)

    # Dikkat haritasını tekrar orijinal x boyutuna büyütüyoruz
    upsample_psi = tf.keras.layers.UpSampling2D(size=(stride_h, stride_w))(sigmoid_xg)

    return tf.keras.layers.multiply([x, upsample_psi])


def attention_unet_olustur(input_size=(128, 128, 3)):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)  # 64x64

    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)  # 32x32

    # Bridge
    b1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)  # 32x32

    # Decoder
    # u1: 32x32 -> 64x64
    u1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b1)
    att1 = attention_gate(x=c2, gating=u1, inter_shape=64)
    merge1 = tf.keras.layers.concatenate([u1, att1])

    # u2: 64x64 -> 128x128
    u2 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(merge1)
    att2 = attention_gate(x=c1, gating=u2, inter_shape=32)
    merge2 = tf.keras.layers.concatenate([u2, att2])

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(merge2)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- 4. EĞİTİM ---
model = attention_unet_olustur()

# Model gelişmeyi bırakırsa eğitimi keser (Early Stopping)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nAttention-Transformer Hibrit Model Eğitimi Başlıyor...")
model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=16,
    callbacks=[callback]
)

# --- 5. KAYDET ---
model.save('deri_analiz_transformer.h5')
print("\nİkinci model başarıyla eğitildi ve 'deri_analiz_transformer.h5' olarak kaydedildi!")