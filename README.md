# Prediksi Keputusan Berlayar

Aplikasi prediksi keputusan berlayar berdasarkan data Wind Speed, Wave Height, Weather, Day of the Week, Boat Technical Condition, menggunakan Flask dan Decision Tree Classifier.

## Fitur:
- Input kondisi Wind Speed,Wave Height,Weather,Day of the Week,Boat Technical Condition
- Prediksi apakah "Berlayar" atau "Tidak Berlayar"
- Menampilkan akurasi model Decision Tree
- Perhitungan Information Gain setiap fitur
- Dataset otomatis diperbarui setelah prediksi

## Teknologi yang Digunakan
- Python
- Flask
- Scikit-Learn (Decision Tree)
- Pandas, NumPy
- HTML

## Cara Menjalankan
1. Clone repository:
     ```bash
   git clone https://github.com/LouisJonathan88/prediksi-keputusan-berlayar.git
   cd prediksi-keputusan-berlayar
   ```
2. Install dependensi:
   ```bash
   pip install flask pandas numpy scikit-learn
   ```

3. Jalankan program:
```bash
python program.py
```
4. Akses di browser:
http://localhost:5000
