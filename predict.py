import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Gunakan model terbaik yang ingin di-deploy.
# Anda bisa mengubah ini ke 'models/mobilenetv2_model.keras' jika ingin menggunakan MobileNetV2
MODEL_PATH = 'models/efficientnetb0_model.keras'

# Mapping index kelas ke nama label (Harus sesuai dengan urutan saat training)
# Karena model di-train menggunakan flow_from_directory, urutan ini biasanya sesuai urutan alfabetis folder.
# Pastikan Anda menyesuaikannya jika folder Anda berbeda.
# Anda juga bisa membaca CLASS_NAMES dari file JSON/pickle jika disave saat training.
# Untuk saat ini, asumsikan kelas berikut (berdasarkan contoh umum dataset sepatu):
# Silakan sesuaikan array ini dengan kelas Anda sebenarnya (misal: ['Ballet Flat', 'Boat', 'Boots', 'Sneakers', dll])
# Kita akan mendeteksi kelas secara dinamis jika mungkin, tapi tanpa file map kita perlu mendeklarasikannya di sini:
try:
    TRAIN_DIR = os.path.join('Shoes Dataset', 'Train')
    CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
except Exception:
    CLASS_NAMES = ['Ballet Flat', 'Boat', 'Brogue', 'Clog', 'Sneaker'] # Placeholder fallback

# Memuat model sekali agar cepat digunakan berulang kali
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"Berhasil memuat model: {MODEL_PATH}")
else:
    print(f"Warning: Model tidak ditemukan di {MODEL_PATH}. Harap jalankan training terlebih dahulu.")

def predict_image(image_path):
    if model is None:
        return "Model Not Found", 0.0

    try:
        # Load and preprocess image
        # Sesuai dengan preprocessing saat training: 224x224
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Rescaling jika menggunakan MobileNetV2 atau jika EfficientNetB0 di-train dengan ImageDataGenerator(rescale=...)
        # Pada file train_efficientnet.py kita MENGHILANGKAN rescale, tapi pada train_mobilenet.py KITA MEMAKAI rescale=1./255.
        # Jadi kita perlu mengecek model apa yang dipakai.
        if 'mobilenet' in MODEL_PATH.lower():
            img_array = img_array / 255.0
            
        # Prediksi
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0
