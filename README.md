# Proyek Regresi Deep Learning: Prediksi Tahun Rilis Lagu

## 1. Ikhtisar Proyek

Proyek ini menerapkan model **Multi-Layer Perceptron (MLP)** menggunakan **TensorFlow/Keras** untuk menyelesaikan tugas regresi. Tujuannya adalah untuk **memprediksi tahun rilis lagu** ('target') berdasarkan 90 fitur numerik yang diekstrak dari properti akustik dan statistik lagu.

Dataset yang digunakan berukuran besar, berisi lebih dari 500.000 sampel. Metodologi yang diterapkan meliputi pra-pemrosesan data, standardisasi, pelatihan model, dan evaluasi menggunakan metrik regresi.

## 2. Dataset

* **Nama File:** `midterm-regresi-dataset.csv`
* **Jumlah Sampel:** 515.345 baris
* **Jumlah Fitur:** 90 fitur numerik.
* **Variabel Target:** Tahun Rilis Lagu (Rata-rata sekitar 1998.40, Standar Deviasi 10.93).
* **Ukuran File:** Sekitar 422.88 MB.

## 3. Metodologi

### 3.1. Pra-pemrosesan Data

1.  **Penentuan Kolom:** Karena data tidak memiliki *header*, kolom pertama ditetapkan sebagai **'target'** (Tahun Rilis) dan 90 kolom berikutnya sebagai **'feature_1'** hingga **'feature_90'**.
2.  **Standardisasi Target:** Variabel target ($y$) distandardisasi menggunakan rumus $y_{scaled} = (y - \mu_y) / \sigma_y$.
3.  **Pembagian Data:** Data dibagi menjadi set pelatihan (*training* 80%) dan set pengujian (*test* 20%).
4.  **Pipeline Fitur:**
    * **Imputasi:** Nilai yang hilang diimputasi menggunakan **median** dari masing-masing fitur (meskipun pada dataset ini tidak ada *missing value* yang eksplisit).
    * **Scaling Fitur:** Seluruh 90 fitur diskalakan menggunakan **StandardScaler** (Z-score normalization) untuk membantu konvergensi model *deep learning*.

### 3.2. Arsitektur Model (MLP)

Model **Sequential** Keras dengan empat lapisan (termasuk lapisan *input* dan *output*):

| Lapisan | Tipe | Jumlah Neuron | Fungsi Aktivasi | Keterangan |
| :--- | :--- | :--- | :--- | :--- |
| Input | Dense | 128 | ReLU | Menerima 90 fitur input. |
| Hidden 1 | Dropout | N/A | N/A | Regularisasi dengan rate **0.2**. |
| Hidden 2 | Dense | 64 | ReLU | |
| Hidden 3 | Dense | 32 | ReLU | |
| Output | Dense | 1 | Linear | Regresi untuk nilai target yang diskalakan. |

* **Total Parameter Model:** 22,017
* **Optimizer:** Adam (Learning Rate = 0.001)
* **Loss Function:** Mean Squared Error (MSE)
* **Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE)

### 3.3. Pelatihan

* **Epochs Maksimum:** 100
* **Batch Size:** 64
* **Validation Split:** 10% dari data pelatihan digunakan untuk validasi.
* **Callback:** **EarlyStopping** diterapkan dengan `patience=5` untuk mencegah *overfitting* dan menghentikan pelatihan lebih awal jika `val_loss` tidak membaik.

## 4. Hasil Evaluasi

Model diuji pada set pengujian (103.069 baris) menggunakan prediksi yang telah **didenormalisasi** kembali ke skala tahun sebenarnya.

| Metrik | Nilai | Keterangan |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | 73.3151 | Rata-rata kuadrat kesalahan. |
| **Root Mean Squared Error (RMSE)** | 8.5624 | Kesalahan prediksi rata-rata dalam satuan tahun. |
| **Mean Absolute Error (MAE)** | **5.8306** | Rata-rata kesalahan mutlak prediksi (error rata-rata adalah **± 5.83 tahun**). |
| **R-squared (R²)** | 0.3840 | Proporsi varian target yang dijelaskan oleh model. |

## 5. Interpretasi

* **MAE 5.83 tahun** menunjukkan bahwa, rata-rata, model memprediksi tahun rilis lagu dengan kesalahan sekitar 5 hingga 6 tahun.
* Nilai **R² 0.3840** (**Moderate Fit** - di bawah ambang batas 0.5) menunjukkan bahwa fitur yang diberikan hanya menjelaskan sebagian kecil hingga sedang dari variabilitas tahun rilis lagu.
* **Rekomendasi:** Diperlukan optimasi lebih lanjut, seperti penyesuaian hyperparameter (jumlah lapisan/neuron, *learning rate*), atau mencoba arsitektur model *deep learning* yang lebih kompleks.
