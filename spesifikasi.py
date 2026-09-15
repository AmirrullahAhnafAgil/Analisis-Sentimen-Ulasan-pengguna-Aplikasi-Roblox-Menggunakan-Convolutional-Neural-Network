import re
import pandas as pd
from tabulate import tabulate

# ============================================================
# 1. KONFIGURASI FILE
# ============================================================
INPUT_FILE = "data/prediksi_all.csv"
OUTPUT_FILE = "data/prediksi_all_spesifikasi.csv"

# ============================================================
# 2. KAMUS KATA KUNCI PER ASPEK/SPESIFIKASI
# ============================================================
ASPEK_KEYWORDS = {
    "Edukasi & Batasan Usia": [
        "roblox kids", "roblox select", "roblox teen", "roblox dewasa",
        "verifikasi umur", "verifikasi usia", "batas umur", "batasan usia",
        "umur", "usia", "anak", "ortu", "orang tua", "dewasa", "remaja",
        "geser umur", "salah umur",
    ],
    "Keamanan & Privasi": [
        "verifikasi wajah", "verifikasi identitas", "scan wajah",
        "privasi", "data pribadi", "keamanan akun", "akun di hack",
        "akun dibobol", "hack", "hacker", "penipu", "phising", "phishing",
        "cheater", "cheat", "hei", "script", "aman",
    ],
    "Fitur & Sosial (Chat/Komunikasi)": [
        "chat", "ngobrol", "komunikasi", "mabar", "teman", "berteman",
        "pertemanan", "voice", "mic", "microphone", "party", "obrolan",
        "japri", "dm ", "add friend", "invite",
    ],
    "Performa & Teknis (Bug/Error)": [
        "bug", "error", "eror", "lag", "ngelag", "lemot", "crash",
        "force close", "fc ", "nge-freeze", "freeze", "not responding",
        "loading", "gagal login", "tidak bisa login", "logout sendiri",
        "update", "server", "koneksi", "internet", "kuota", "install",
        "download", "hapus aplikasi", "uninstall", "iklan", "ads",
    ],
    "Konten & Gameplay": [
        "map", "game nya", "permainan", "grafis", "tampilan", "level",
        "tower", "obby", "brainrot", "adopt me", "murder mystery",
        "item", "skin", "avatar", "seru", "menarik", "fitur baru",
        "mode", "event",
    ],
}

# Urutan prioritas pengecekan aspek (dari paling spesifik ke paling umum)
URUTAN_PRIORITAS = [
    "Edukasi & Batasan Usia",
    "Keamanan & Privasi",
    "Fitur & Sosial (Chat/Komunikasi)",
    "Performa & Teknis (Bug/Error)",
    "Konten & Gameplay",
]

ASPEK_LAINNYA = "Umum/Lainnya"


def deteksi_spesifikasi(teks: str) -> str:
    """
    Mendeteksi aspek/spesifikasi dari satu baris ulasan berdasarkan
    kemunculan kata kunci. Mengembalikan nama aspek pertama yang cocok
    sesuai urutan prioritas. Jika tidak ada kata kunci yang cocok,
    dikembalikan "Umum/Lainnya".
    """
    if not isinstance(teks, str) or teks.strip() == "":
        return ASPEK_LAINNYA

    teks_lower = teks.lower()

    for aspek in URUTAN_PRIORITAS:
        kata_kunci_list = ASPEK_KEYWORDS[aspek]
        for kata_kunci in kata_kunci_list:
            # gunakan word boundary sederhana lewat regex agar lebih presisi
            pola = re.escape(kata_kunci)
            if re.search(pola, teks_lower):
                return aspek

    return ASPEK_LAINNYA


def main():
    # --------------------------------------------------------
    # 1. Baca file prediksi_all.csv
    # --------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    kolom_asli = ["text", "true", "pred", "confidence", "status"]
    kolom_asli = [k for k in kolom_asli if k in df.columns]

    print(f"Jumlah baris terbaca dari {INPUT_FILE}: {len(df)}")
    print(f"Kolom asli: {kolom_asli}\n")

    # --------------------------------------------------------
    # 2. Tambahkan kolom spesifikasi berdasarkan teks ulasan
    # --------------------------------------------------------
    df["spesifikasi"] = df["text"].apply(deteksi_spesifikasi)

    # --------------------------------------------------------
    # 3. Simpan ke file BARU
    # --------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"File baru berhasil disimpan: {OUTPUT_FILE}\n")

if __name__ == "__main__":
    main()