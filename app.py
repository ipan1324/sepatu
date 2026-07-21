from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from predict import predict_image, MODEL_PATH

app = Flask(__name__)
app.secret_key = 'capstone_super_secret_key'

# Konfigurasi Upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder uploads ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simulasi database sederhana (riwayat prediksi)
# Pada production, gunakan SQLite/PostgreSQL
history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Periksa apakah ada file di request
        if 'file' not in request.files:
            flash('Tidak ada file yang diunggah.', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Periksa apakah file kosong
        if file.filename == '':
            flash('Tidak ada gambar yang dipilih.', 'warning')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Amankan nama file dan beri UUID agar unik
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Melakukan prediksi menggunakan modul predict.py
            pred_class, confidence = predict_image(filepath)
            
            # Mendapatkan nama model dari MODEL_PATH
            model_name = os.path.basename(MODEL_PATH)
            
            if pred_class == "Error":
                flash('Terjadi kesalahan saat memproses gambar.', 'danger')
                return redirect(request.url)
                
            # Simpan riwayat
            history_entry = {
                'filename': unique_filename,
                'class': pred_class,
                'confidence': f"{confidence:.2f}",
                'model': model_name
            }
            history.insert(0, history_entry)
            
            # Redirect ke halaman hasil
            return render_template('result.html', result=history_entry)
        else:
            flash('Format file tidak diizinkan. Gunakan JPG, JPEG, PNG, atau WEBP.', 'danger')
            return redirect(request.url)

    return render_template('predict.html')

@app.route('/history')
def get_history():
    # Mengembalikan riwayat prediksi dalam bentuk JSON
    return jsonify(history)

# Menambahkan route statis untuk menampilkan gambar unggahan di result.html
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Menjalankan Web Server...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
