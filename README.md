# UTS-DATA-MINING-2304020014
**1. LATAR BELAKANG**

Perkembangan teknologi machine learning telah memberikan banyak manfaat dalam proses analisis data, termasuk dalam bidang industri makanan dan minuman. Salah satu penerapan machine learning adalah dalam memprediksi kualitas wine berdasarkan karakteristik kimia yang dimiliki. Kualitas wine dipengaruhi oleh berbagai faktor seperti kadar alkohol, tingkat keasaman, kandungan gula, dan senyawa kimia lainnya sehingga diperlukan metode yang mampu menganalisis hubungan antar variabel secara efektif.

Proses penilaian kualitas wine secara manual memerlukan waktu yang cukup lama dan bergantung pada penilaian subjektif seseorang. Oleh karena itu, penggunaan algoritma machine learning dapat membantu melakukan prediksi kualitas wine secara otomatis dengan memanfaatkan data historis yang tersedia. Pada penelitian ini digunakan algoritma Random Forest untuk membangun model klasifikasi kualitas wine karena algoritma tersebut mampu menangani data numerik dengan baik serta memiliki performa yang stabil dalam berbagai kondisi data.

**2. RUMUSAN MASALAH**

Berdasarkan latar belakang yang telah dijelaskan, maka penelitian ini berfokus pada bagaimana membangun model machine learning yang mampu memprediksi kualitas wine berdasarkan karakteristik kimia yang dimiliki, mengetahui performa algoritma Random Forest dan Decision Tree dalam melakukan klasifikasi kualitas wine, serta mengetahui variabel yang paling berpengaruh terhadap hasil prediksi kualitas wine.

**3. TUJUAN**

Penelitian ini bertujuan untuk membangun model machine learning dalam memprediksi kualitas wine menggunakan algoritma Random Forest dan Decision Tree, mengevaluasi performa kedua model klasifikasi tersebut, serta mengetahui variabel yang paling berpengaruh terhadap kualitas wine.

**4. DATASET DAN VARIABEL**

Dataset yang digunakan dalam penelitian ini merupakan dataset kualitas wine yang terdiri dari beberapa variabel numerik yang berkaitan dengan karakteristik kimia wine. Dataset dibagi menjadi dua bagian, yaitu data training untuk melatih model dan data testing untuk melakukan prediksi.

Variabel yang digunakan pada dataset antara lain:

| Variabel | Keterangan |
|---|---|
| fixed acidity | Tingkat keasaman tetap |
| volatile acidity | Tingkat keasaman mudah menguap |
| citric acid | Kandungan asam sitrat |
| residual sugar | Kandungan gula sisa |
| chlorides | Kandungan klorida |
| free sulfur dioxide | Sulfur dioksida bebas |
| total sulfur dioxide | Total sulfur dioksida |
| density | Kepadatan wine |
| pH | Tingkat keasaman |
| sulphates | Kandungan sulfat |
| alcohol | Kandungan alkohol |
| quality | Kualitas wine (target) |

**5. ALUR PENELITIAN**

Tahapan penelitian yang dilakukan pada penelitian ini adalah sebagai berikut:

1. Persiapan data
2. Membaca dataset
3. Pembersihan data
4. Pemisahan fitur dan target
5. Pembagian data training dan validasi
6. Pembuatan model Random Forest
7. Evaluasi model
8. Visualisasi feature importance
9. Prediksi data testing
10. Penyimpanan hasil prediksi

**6. PERSIAPAN DATA**

Tahap pertama dilakukan dengan mengimpor berbagai library yang diperlukan dalam proses pengolahan data, visualisasi, pembuatan model, serta evaluasi model machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Evaluasi Model
from sklearn.metrics import accuracy_score, classification_report
```

Library pandas dan numpy digunakan untuk pengolahan data, matplotlib dan seaborn digunakan untuk visualisasi data, sedangkan library dari scikit-learn digunakan untuk membangun serta mengevaluasi model machine learning.

**7. Membaca Dataset**

Dataset training dan testing dibaca menggunakan fungsi `read_csv()`

```python
train = pd.read_csv('/content/data_training.csv')
test = pd.read_csv('/content/data_testing.csv')
```

Selanjutnya dilakukan pengecakan beberapa baris pertama dataset menggunakan fungsi `head()`

```python
train.head()
test.head()
```

Fungsi tersebut digunakan untuk melihat struktur data, nama kolom, serta isi awal dataset

**8. PENGECEKAN INFORMASI DATASET**

Pada penelitian ini, informasi umum pada dataset ditampilkan menggunakan fungsi `info()` dan `describe()`

```python
train.info()
train.describe()
```

Fungsi `info()` digunakan untuk melihat tipe data, jumlah data, serta untuk cek keberadaan missing value, sedangkan `describe()` digunakan untuk melihat statistik data seperti rata-rata, nilai minimum, nimai maksimum, dan standar deviasi.
