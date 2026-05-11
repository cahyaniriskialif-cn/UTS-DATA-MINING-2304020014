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

**9. PEMBERSIHAN DATA**

Pada tahap ini dilakukan pengecekan missing value untuk memastikan tidak terdapat data kosong pada dataset

```python
print("Missing Train:\n", train.isnull().sum())
print("\nMissing Test:\n", test.isnull().sum())
```

Berdasarkan hasil pengecekan, dataset tidak memiliki missing value sehingga tidak diperlukan proses penanganan data kosong.

Selanjutnya kolom `Id` dihapus karena hanya berfungsi sebagai identitas data dan tidak berpengaruh terhadap proses prediksi

```python
train = train.drop('Id', axis=1)

test_id = test['Id']

test = test.drop('Id', axis=1)
```

Kolom `Id pada data testing disimpan terlebih dahulu agar dapat digunakan kembali pada proses penyusunan hasil prediksi.

**10. PEMISAHAN FITUR DAN TARGET**

Pada tahap ini dilakukan pemisihan fitur dan target dengan menggunakan kode
```python
X = train.drop('quality', axis=1)
y = train['quality']
```
Variabel `x` berisi seluruh fitur yang digunakan sebagai input model, sedangkan variabel `y` berisi target berupa kualitas wine yang akan diprediksi.

**11. PEMBAGIAN DATA TRAINING DAN VALIDASI**

Pada tahapan ini dataset dibagi menjadi data training dan data validasi menggunakan fungsi `train_test_split().

```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```
Pada tahap ini, sebanyak 80% data digunakan untuk melatih model dan 20% digunakan untuk evaluasi model. Parameter `random_state=42` digunakan agar proses pembagian data tetap konsisten setiap kali program dijalankan.

**12. PEMBUATAN MODEL RANDOM FOREST**

Pada tahap ini, model pertama yang digunakan dalam penelitian ini adalah Random Forest.
```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)
```
Algoritma Random Forest dipilih karena mampu menangani data numerik dengan baik dan memiliki performa yang stabil dalam proses klasifikasi. Parameter `class_weight='balanced' digunakan untuk menangani ketidakseimbangan jumlah data antar kelas. Selain itu, fungsi `fit()` digunakan untuk melatih model menggunakan data training sehingga model dapat mempelajari pola hubungan antara fitur dan target.

**13. EVALUASI MODEL RANDOM FOREST**

Pada tahapan ini, setelah model selesai dilatih, dilakukan proses evaluasi menggunakan data validasi.

```python
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("
Classification Report:
", classification_report(y_val, y_pred))
```

Evaluasi dilakukan menggunakan nilai accuracy serta classification report yang terdiri dari precision, recall, dan f1-score. Berdasarkan hasil evaluasi, model Random Forest memperoleh akurasi sekitar 57%–60%. Hasil tersebut menunjukkan bahwa model cukup baik dalam memprediksi kelas mayoritas, namun masih mengalami kesulitan dalam memprediksi kelas minoritas karena jumlah data yang tidak seimbang.

**14. PEMBUATAN MODEL DECISION TREE**

Selain Random Forest, penelitian ini juga menggunakan algoritma Decision Tree untuk membandingkan performa model klasifikasi. 

```python
dt = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

dt.fit(X_train, y_train)
```
Decision Tree merupakan algoritma klasifikasi yang bekerja dengan membentuk struktur pohon keputusan berdasarkan fitur-fitur yang digunakan. Algoritma ini dipilih karena mudah dipahami dan mampu menghasilkan aturan klasifikasi yang jelas.

**15. EVALUASI MODEL DECISION TREE**

Pada tahap ini Model Decision Tree dievaluasi menggunakan data validasi.
```python
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, dt_pred))
print("
Classification Report:
", classification_report(y_val, dt_pred))
```

Berdasarkan hasil evaluasi, model Decision Tree mampu melakukan klasifikasi kualitas wine dengan performa yang cukup baik, meskipun hasil akurasinya masih sedikit di bawah Random Forest. Hal ini terjadi karena Random Forest merupakan pengembangan dari Decision Tree yang menggunakan banyak pohon keputusan sehingga mampu menghasilkan prediksi yang lebih stabil.

Selain menggunakan accuracy dan classification report, evaluasi model juga dilakukan menggunakan confusion matrix untuk melihat jumlah prediksi yang benar dan salah pada setiap kelas.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, dt_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Decision Tree')

plt.show()
```
Confusion matrix digunakan untuk melihat performa model secara lebih detail berdasarkan jumlah prediksi yang sesuai dan tidak sesuai pada masing-masing kelas kualitas wine. Nilai diagonal pada confusion matrix menunjukkan jumlah prediksi yang benar, sedangkan nilai di luar diagonal menunjukkan kesalahan prediksi model.

**16. VISUALISASAI DECISION TREE**

Pada tahap ini Visualisasi Decision Tree dilakukan untuk melihat struktur pohon keputusan yang terbentuk pada model.

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))

plot_tree(
    dt,
    feature_names=X.columns,
    class_names=True,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.show()
```
Visualisasi pohon keputusan menunjukkan bagaimana model melakukan proses klasifikasi berdasarkan nilai fitur tertentu. Setiap node pada pohon berisi aturan keputusan yang digunakan model untuk membagi data ke dalam kelas tertentu. Warna pada node menunjukkan kelas dominan yang diprediksi oleh model.

Melalui visualisasi ini dapat diketahui fitur-fitur yang sering digunakan dalam proses pengambilan keputusan, sehingga membantu memahami cara kerja model Decision Tree secara lebih jelas.

**17. VISUALISASI FE

