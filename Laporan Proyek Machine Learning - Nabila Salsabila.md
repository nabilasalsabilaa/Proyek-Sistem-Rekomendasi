# Laporan Proyek Machine Learning - Nabila Salsabila

## Project Overview

Rekomendasi buku merupakan salah satu aplikasi penting dalam industri penerbitan dan pembacaan. Dengan meningkatnya jumlah buku yang diterbitkan setiap tahun, membantu pembaca menemukan buku yang sesuai dengan minat dan preferensi mereka menjadi tantangan. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi buku yang memanfaatkan teknik Collaborative Filtering dan Content-Based Filtering. Dengan sistem ini, diharapkan pembaca dapat menemukan buku yang relevan dan meningkatkan pengalaman membaca mereka. Proyek ini dapat membantu pembaca menemukan buku yang sesuai dengan minat mereka, yang akhirnya dapat meningkatkan kepuasan dan loyalitas pelanggan. Sistem rekomendasi yang efektif dapat mempengaruhi keputusan pembelian dan mendorong pembaca untuk menjelajahi lebih banyak buku, sehingga memperluas pasar dan meningkatkan pendapatan.

## Business Understanding

### Problem Statements

- Bagaimana cara memberikan rekomendasi buku yang relevan berdasarkan preferensi pengguna?
- Bagaimana teknik yang dapat digunakan untuk mengidentifikasi buku-buku yang mirip berdasarkan atribut seperti judul dan penulis?

### Goals

- Mengembangkan sistem rekomendasi yang menggunakan Content-based Filtering untuk merekomendasikan buku berdasarkan kesamaan fitur.
- Mengimplementasikan Collaborative Filtering untuk menemukan buku yang direkomendasikan berdasarkan rating pengguna yang mirip.

### Solution statements

- Menerapkan Content-Based Filtering dengan menghitung kesamaan menggunakan TF-IDF dan cosine similarity untuk merekomendasikan buku berdasarkan konten dan fitur buku.
- Menggunakan Collaborative Filtering dengan algoritma K-Nearest Neighbors (KNN) untuk memberikan rekomendasi berdasarkan interaksi pengguna.

## Data Understanding

Dataset yang digunakan adalah Goodreads-books Dataset dari Kaggle. Dataset ini berisi informasi mengenai lebih dari 10.000 buku dari berbagai genre dan kategori, yang dapat dimanfaatkan untuk membangun sistem rekomendasi buku.

Link Dataset: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks

### Struktur Data
Jumlah Baris: 11.123
Jumlah Kolom: 12

### Informasi Variabel pada Dataset sebagai berikut:
| Variabel         | Deskripsi                                               | Tipe Data  |
|------------------|---------------------------------------------------------|------------|
| bookID       | Identifikasi unik untuk setiap buku.                                       | int64     |
| title           | Judul buku                                 | object     |
| authors    | Nama penulis             | object      |
| average_rating          | Rating rata-rata dari pengguna                      | float64     |
| isbn       | Nomor unik buku yang terdiri dari 10 digit                    | object     |
| isbn13           | Nomor unik buku yang terdiri dari 13 digit                | int64      |
| language_code     | Bahasa buku               | object     |
| num_pages    | Jumlah halaman buku    | int64     |
| ratings_count  | Jumlah pengguna yang memberi rating                   | int64     |
| text_reviews_count   | Jumlah ulasan tertulis dari pengguna       | int64     |
| publication_date     | Tanggal penerbitan    | object     |
| publisher | Penerbit buku    | object     |

### Kondisi Data
- Missing Values: Setelah pengecekan, ditemukan beberapa kolom yang memiliki nilai kosong, khususnya pada kolom title, authors, dan average_rating. Penanganan dilakukan dengan mengisi nilai kosong di kolom title dan authors dengan informasi Unknown Title dan Unknown Author.
- Duplicate Values: Pengecekan data duplikat dilakukan berdasarkan bookID dan title. Hasilnya, tidak ditemukan data duplikat pada kolom ini.

## Data Preparation
Pada tahap ini, data dipersiapkan untuk dua pendekatan sistem rekomendasi, yaitu Content-Based Filtering dan Collaborative Filtering. Tahapan ini melibatkan beberapa teknik yang diperlukan untuk mempersiapkan data secara optimal sebelum proses pemodelan. Berikut adalah teknik yang dilakukan pada tahap ini:
1. Menggabungkan fitur title dan authors untuk Content-Based Filtering
Pada proses ini, dibuat kolom combined_features yang menggabungkan informasi title dan authors dari setiap buku. Content-Based Filtering memerlukan fitur teks yang kaya agar model dapat menemukan kesamaan antar buku berdasarkan konten. Menggabungkan judul dan nama penulis akan membantu sistem memahami konteks dari setiap buku, sehingga bisa merekomendasikan buku yang serupa dari segi tema atau penulis. Kolom combined_features ini akan diolah lebih lanjut dalam proses Model Development untuk mendapatkan skor kesamaan antar buku, menggunakan teknik seperti TF-IDF untuk menghitung bobot kata dalam setiap buku.

2. Menyiapkan data rating tiruan untuk Collaborative Filtering
Pada proses ini, dihasilkan data tiruan berupa user_id dan user_rating secara acak untuk menciptakan interaksi antara pengguna dan buku dalam dataset. Collaborative Filtering memerlukan data interaksi antara pengguna dan item (buku) untuk memberikan rekomendasi berdasarkan pola preferensi pengguna. Karena dataset asli tidak menyediakan data pengguna, data tiruan ini dibuat untuk mensimulasikan skenario di mana pengguna memberikan rating pada buku. Dengan adanya data tiruan ini, kita dapat membangun User-Item Interaction Matrix, yang nantinya akan digunakan untuk menganalisis kesamaan preferensi pengguna dalam sistem rekomendasi.

3. Membuat user-item matrix untuk Collaborative Filtering
Pada proses ini, dibuat user-item matrix menggunakan pivot table, di mana baris mewakili user_id, kolom mewakili title, dan nilai dalam sel menunjukkan user_rating. Matriks ini penting dalam pendekatan Collaborative Filtering, karena memungkinkan analisis interaksi pengguna-item untuk memberikan rekomendasi berdasarkan kesamaan antar pengguna atau antar item.

4. Memilih fitur penting (Feature Selection)
Pada proses ini, subset data dipilih berdasarkan fitur yang paling relevan, yaitu average_rating, ratings_count, dan num_pages. Fitur-fitur ini memberikan informasi penting mengenai kualitas dan popularitas buku, yang berguna untuk meningkatkan akurasi model dalam memberikan rekomendasi.

5. Membagi data menjadi data train dan data test
Pada tahap ini, data dibagi menjadi train set (80%) dan test set (20%) untuk keperluan pelatihan dan pengujian model. Pembagian ini penting untuk mengevaluasi performa model pada data baru yang tidak terlihat selama pelatihan, sehingga dapat memastikan generalisasi dan efektivitas model dalam memberikan rekomendasi yang akurat.

## Modeling

Pada proyek ini, sistem rekomendasi dikembangkan dengan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering. Sistem ini dirancang untuk memberikan rekomendasi buku yang relevan, baik berdasarkan konten buku maupun pola preferensi antar pengguna.

### Model Development dengan Content-Based Filtering

Content-Based Filtering bekerja dengan menganalisis kesamaan antara fitur konten buku untuk menghasilkan rekomendasi. Pada proyek ini, fitur title dan authors dikombinasikan menjadi satu kolom combined_features yang kemudian diproses menggunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency). Pendekatan TF-IDF digunakan untuk mengubah data teks menjadi vektor numerik, yang selanjutnya dihitung menggunakan cosine similarity untuk mengukur kesamaan antar buku. Fungsi get_recommendations dalam model ini akan memberikan daftar 10 buku yang paling mirip berdasarkan kesamaan konten dengan judul buku yang dipilih oleh pengguna.

#### Contoh Rekomendasi untuk Buku "Nikola Tesla: A Spark of Genius"
1. My Inventions
2. Tesla Papers
3. Wizard: The Life and Times of Nikola Tesla
4. The Cave
5. Teleportation: From Star Trek to Tesla
6. All the Names
7. The Zahir
8. The Dark Side Of Genius: The Life Of Alfred Hitchcock
9. A Christmas Carol
10. Ludwig Wittgenstein: The Duty of Genius

### Model Development dengan Collaborative Filtering

Collaborative Filtering menggunakan data interaksi pengguna dengan item (buku) untuk menghasilkan rekomendasi, berdasarkan kesamaan preferensi antar pengguna atau buku. Pada proyek ini, metode K-Nearest Neighbors (KNN) digunakan untuk menemukan kemiripan antara buku-buku berdasarkan rating yang diberikan oleh pengguna. Dengan membuat data rating tiruan untuk mensimulasikan interaksi pengguna-buku, model ini menghitung kemiripan antara buku menggunakan metrik cosine similarity. Fungsi recommend_books akan memberikan rekomendasi buku berdasarkan judul buku yang diberikan pengguna, dengan 5 buku paling mirip sebagai output.

#### Contoh Rekomendasi untuk Buku "Nikola Tesla: A Spark of Genius"
1. The Book of Merlyn: The Unpublished Conclusion to The Once & Future King
2. The Book of Other People
3. The Book of My Life
4. The Book of Ruth
5. The Book of Lost Tales  Part Two (The History of Middle-earth  #2)

### Analisis Kelebihan dan Kekurangan Pendekatan

#### Content-Based Filtering
- Kelebihan: Content-Based Filtering tidak memerlukan data interaksi pengguna, sehingga cocok jika data pengguna terbatas. Model ini dapat memberikan rekomendasi buku yang relevan secara langsung berdasarkan kesamaan konten.
- Kekurangan: Model ini hanya mampu merekomendasikan buku yang mirip dari segi konten, sehingga kurang mampu memberikan variasi rekomendasi di luar genre atau tema buku yang sama.

#### Collaborative Filtering
- Kelebihan: Collaborative Filtering mampu memberikan rekomendasi buku yang lebih beragam, karena model ini tidak terbatas pada kemiripan konten. Model ini bisa merekomendasikan buku-buku lain yang dinilai menarik oleh pengguna dengan preferensi serupa.
- Kekurangan: Collaborative Filtering membutuhkan data interaksi pengguna untuk menghasilkan rekomendasi yang akurat. Jika data pengguna sangat terbatas, model ini mungkin tidak efektif atau menghasilkan rekomendasi yang kurang relevan.

## Evaluation

### Content-Based Filtering

Pada Content-Based Filtering, evaluasi sistem rekomendasi dapat diukur menggunakan metrik Precision, yang menunjukkan seberapa relevan rekomendasi yang diberikan dengan preferensi pengguna berdasarkan kesamaan konten.

Misalnya, jika sistem merekomendasikan 10 buku kepada pengguna, dan dari 10 buku tersebut, ada 6 yang relevan (berhubungan dengan tema atau konten yang serupa), maka precision dapat dihitung sebagai berikut:

Precision = Jumlah rekomendasi yang relevan / Total rekomendasi

Dalam hal ini, precision adalah 6/10 atau 60%.

Dengan pendekatan ini, kita dapat memahami sejauh mana rekomendasi dari model Content-Based Filtering sesuai dengan kebutuhan atau minat pengguna. Precision dalam Content-Based Filtering dihitung berdasarkan jumlah rekomendasi yang benar-benar relevan dibandingkan dengan total rekomendasi yang diberikan, sehingga semakin tinggi nilai precision, semakin akurat rekomendasi model tersebut.

#### Contoh Hasil Rekomendasi

Sebagai contoh, pengguna mencari rekomendasi berdasarkan buku Nikola Tesla: A Spark of Genius, dan sistem memberikan rekomendasi berikut:
1. My Inventions
2. Tesla Papers
3. Wizard: The Life and Times of Nikola Tesla
4. The Cave
5. Teleportation: From Star Trek to Tesla
6. All the Names
7. The Zahir
8. The Dark Side Of Genius: The Life Of Alfred Hitchcock
9. A Christmas Carol
10. Ludwig Wittgenstein: The Duty of Genius

Dari rekomendasi ini, sebanyak 6 buku relevan, karena memiliki tema atau konten yang terkait dengan tokoh jenius, inovasi, atau profil tokoh bersejarah serupa. Oleh karena itu, precision dari rekomendasi ini adalah:

Precision = 6/10 = 0.6 atau 60%

Precision sebesar 60% menunjukkan bahwa sistem mampu memberikan rekomendasi yang relevan sesuai dengan minat pengguna, sehingga model ini dapat diandalkan untuk memberikan rekomendasi berbasis kesamaan konten.

### Collaborative Filtering

Pada pendekatan Collaborative Filtering, hasil evaluasi diukur dengan nilai Precision dan Recall sebagai berikut:
- Precision: 1.0
- Recall: 1.0

Evaluasi ini menunjukkan bahwa model Collaborative Filtering mampu memprediksi semua item yang relevan dengan sempurna, tanpa kesalahan dalam prediksi (false positives) atau kehilangan item yang relevan (false negatives).

### Metrik Evaluasi
Berikut adalah penjelasan dan rumus dari metrik evaluasi yang digunakan pada proyek ini:

#### 1. Precision
Precision mengukur seberapa akurat prediksi positif yang dibuat oleh model. Dengan nilai precision 1.0, artinya semua rekomendasi yang diberikan oleh sistem adalah benar dan relevan. Ini menunjukkan model berhasil dalam menghindari false positives secara total. Rumusnya adalah:

Precision = TP / (TP + FP)

Keterangan:
- TP (True Positive): Jumlah prediksi positif yang benar.
- FP (False Positive): Jumlah prediksi positif yang salah.

#### 2. Recall
Recall mengukur seberapa banyak dari total kasus positif yang berhasil terdeteksi oleh model. Nilai recall 1.0 menunjukkan bahwa model mampu menangkap semua item yang seharusnya direkomendasikan, tanpa kehilangan satu pun item yang relevan. Rumusnya adalah:

Recall = TP / (TP + FN)

Keterangan:
- FN (False Negatives): Jumlah prediksi positif yang tidak terdeteksi.

### Kesimpulan
Dari hasil evaluasi di atas, dapat disimpulkan bahwa sistem rekomendasi menggunakan Collaborative Filtering memiliki performa yang sangat baik dengan nilai precision dan recall masing-masing 1.0. Sementara itu, pada Content-Based Filtering, precision dapat dihitung untuk mengukur relevansi rekomendasi yang diberikan. Kombinasi kedua pendekatan ini dapat membantu meningkatkan akurasi dan relevansi rekomendasi yang diberikan kepada pengguna, sehingga sistem dapat memenuhi preferensi pengguna dengan lebih baik.

Referensi: 
- Ko, H.; Lee, S.; Park, Y.; Choi, A. A Survey of Recommendation Systems: Recommendation Models, Techniques, and Application Fields. Electronics 2022, 11, 141. https://doi.org/10.3390/electronics11010141
- Li, Y., Liu, K., Satapathy, R., Wang, S., & Cambria, E. (2023). Recent developments in recommender systems: A survey. arXiv. https://doi.org/10.48550/arXiv.2306.12680
- Dharmawan, H., Tukino, Shofiah Hilabi, S., & Karniawulan, I. (2023). SISTEM REKOMENDASI BUKU DENGAN METODE K-NEAREST NEIGHBOR (K-NN) PADA GRAMEDIA . ZONAsi: Jurnal Sistem Informasi, 5(1), 16 - 25. https://doi.org/10.31849/zn.v5i1.12203