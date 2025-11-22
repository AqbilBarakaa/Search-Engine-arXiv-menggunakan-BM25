from flask import Flask, render_template, request  # Untuk membuat web app dan menerima input user
from bm25 import getDataFromJson, organize, retrieve  # Import fungsi utama dari utils.py (data init dan retrieval)
import os  # Untuk memeriksa keberadaan file

app = Flask(__name__)  # Inisialisasi aplikasi Flask

# --- ROUTE UTAMA (HALAMAN BERANDA) ---

@app.route("/", methods=["GET", "POST"])
def index():
    results = []  # Menampung hasil pencarian
    query = ""  # Menyimpan query dari pengguna

    if request.method == "POST":
        query = request.form["query"]  # Ambil input query dari form HTML
        results = retrieve(query)  # Jalankan fungsi retrieve untuk ambil hasil pencarian dengan BM25

    # Render halaman index.html dan kirimkan query serta hasil pencarian ke dalam template
    return render_template("index.html", query=query, results=results)

# --- JALANKAN APLIKASI ---

if __name__ == "__main__":
    # Cek apakah file hasil preprocessing sudah ada, kalau belum lakukan preprocessing
    if not os.path.exists("full_data_processed_FINAL.p"):
        print("Processing data...")  # Log bahwa data sedang diproses
        getDataFromJson()  # Membaca dan preprocessing data mentah dari JSON
        organize()  # Membuat keyword dan inverted index

    # Jalankan aplikasi Flask dalam mode debug
    app.run(debug=True)
