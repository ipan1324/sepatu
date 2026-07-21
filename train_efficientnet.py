import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode - tidak butuh GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import label_binarize

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = 'Shoes Dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
VALID_DIR = os.path.join(BASE_DIR, 'Valid')
TEST_DIR = os.path.join(BASE_DIR, 'Test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'models/efficientnetb0_model.keras'
HISTORY_SAVE_PATH = 'models/efficientnetb0_history.npy'

# Pastikan folder models/ ada
os.makedirs('models', exist_ok=True)

# ==============================================================================
# PREPROCESSING & DATA AUGMENTATION
# ==============================================================================
# EfficientNet di Keras mengharapkan input tanpa rescale, karena rescale sudah 
# ada di dalam layer arsitektur EfficientNet secara otomatis jika dipanggil melalui keras.
# Namun, untuk konsistensi dengan prompt yang meminta normalisasi, dan menghindari
# error komputasi yang ganda, kita tidak menambahkan rescale untuk aug generator.
# Catatan: Jika menggunakan tf.keras.applications.EfficientNetB0, normalisasinya 
# sudah dimasukkan (built-in). Jadi kita hapus rescale=1./255.
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"Ditemukan {NUM_CLASSES} kelas: {CLASS_NAMES}")

# ==============================================================================
# MODELING: EfficientNetB0
# ==============================================================================
def create_model():
    # Load EfficientNetB0 dengan pre-trained weights ImageNet, tanpa layer classifier paling atas
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze feature extractor
    base_model.trainable = False

    # Tambahkan custom classifier di atas base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Gabungkan menjadi satu model utuh
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model()
model.summary()

# ==============================================================================
# TRAINING CALLBACKS
# ==============================================================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

# ==============================================================================
# TRAINING THE MODEL
# ==============================================================================
print("Memulai pelatihan model EfficientNetB0...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr, model_checkpoint]
)

# Simpan history
np.save(HISTORY_SAVE_PATH, history.history)

# ==============================================================================
# EVALUATION & VISUALIZATION
# ==============================================================================
# 1. Plot Accuracy & Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('models/efficientnetb0_training_plot.png')
plt.close()

# 2. Evaluasi pada Test Set
print("\nMengevaluasi model pada Test Dataset...")
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Hitung Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - EfficientNetB0')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('models/efficientnetb0_confusion_matrix.png')
plt.close()

# ROC Curve
try:
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", NUM_CLASSES)
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {CLASS_NAMES[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - EfficientNetB0')
    plt.legend(loc="lower right")
    plt.savefig('models/efficientnetb0_roc_curve.png')
    plt.close()
except Exception as e:
    print(f"Tidak dapat membuat plot ROC: {e}")

print(f"Model berhasil disimpan di: {MODEL_SAVE_PATH}")
