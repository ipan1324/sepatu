import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to saved histories
MN_HISTORY_PATH = 'models/mobilenetv2_history.npy'
EN_HISTORY_PATH = 'models/efficientnetb0_history.npy'

def plot_comparison():
    if not os.path.exists(MN_HISTORY_PATH) or not os.path.exists(EN_HISTORY_PATH):
        print("Satu atau kedua file history tidak ditemukan. Pastikan Anda telah melatih kedua model (train_mobilenet.py dan train_efficientnet.py).")
        return

    # Load history data
    mn_hist = np.load(MN_HISTORY_PATH, allow_pickle=True).item()
    en_hist = np.load(EN_HISTORY_PATH, allow_pickle=True).item()

    plt.figure(figsize=(14, 6))

    # Compare Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(mn_hist['val_accuracy'], label='MobileNetV2 Val Acc', color='blue', linestyle='--')
    plt.plot(en_hist['val_accuracy'], label='EfficientNetB0 Val Acc', color='green')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Compare Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(mn_hist['val_loss'], label='MobileNetV2 Val Loss', color='blue', linestyle='--')
    plt.plot(en_hist['val_loss'], label='EfficientNetB0 Val Loss', color='green')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/models_comparison_plot.png')
    plt.show()

    # Menampilkan tabel perbandingan metrik akhir
    mn_best_acc = max(mn_hist['val_accuracy'])
    en_best_acc = max(en_hist['val_accuracy'])
    mn_best_loss = min(mn_hist['val_loss'])
    en_best_loss = min(en_hist['val_loss'])

    print("==========================================================")
    print("                HASIL PERBANDINGAN MODEL                  ")
    print("==========================================================")
    print(f"{'Metrik':<20} | {'MobileNetV2':<15} | {'EfficientNetB0':<15}")
    print("-" * 58)
    print(f"{'Best Val Accuracy':<20} | {mn_best_acc:<15.4f} | {en_best_acc:<15.4f}")
    print(f"{'Best Val Loss':<20} | {mn_best_loss:<15.4f} | {en_best_loss:<15.4f}")
    print("==========================================================\n")
    
    print("Analisis Singkat:")
    print("- Kecepatan Training: MobileNetV2 umumnya memiliki waktu training per epoch yang lebih cepat.")
    print("- Jumlah Parameter: MobileNetV2 (~3.4M parameter) vs EfficientNetB0 (~5.3M parameter).")
    print("- Kelebihan MobileNetV2: Sangat ringan, inferensi cepat, cocok untuk perangkat seluler / edge.")
    print("- Kelebihan EfficientNetB0: Akurasi biasanya lebih tinggi berkat metode compound scaling, ekstraksi fitur lebih kaya.")
    print("- Kekurangan: MobileNetV2 mungkin kalah akurat di dataset kompleks, sedangkan EfficientNetB0 memakan memori sedikit lebih besar.")

if __name__ == '__main__':
    plot_comparison()
