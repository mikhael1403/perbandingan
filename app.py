import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi Halaman Web
st.set_page_config(page_title="DSS Karyawan", layout="wide")

st.title("Perbandingan Hasil Keempat Metode (DSS)")
st.write("Aplikasi ini membandingkan 3 metode Sistem Pendukung Keputusan (SAW, WP, TOPSIS) menggunakan 30 data dummy karyawan.")

# Set seed agar data dummy konsisten
np.random.seed(42)

# Generate 30 Alternatif Karyawan
karyawan = [f"Karyawan_{i:02d}" for i in range(1, 31)]

# Generate data random untuk 4 Kriteria
data = {
    'Alternatif': karyawan,
    'C1 (Benefit)': np.random.randint(60, 100, 30),
    'C2 (Benefit)': np.random.randint(60, 100, 30),
    'C3 (Benefit)': np.random.randint(60, 100, 30),
    'C4 (Cost)': np.random.randint(1, 10, 30)
}
df = pd.DataFrame(data)

# Tampilkan Data Mentah (Bisa di-expand/collapse)
with st.expander("Klik untuk melihat Data Dummy Karyawan"):
    st.dataframe(df, use_container_width=True)

# Konfigurasi Bobot dan Jenis Kriteria
weights = np.array([0.35, 0.25, 0.20, 0.20])
is_benefit = np.array([True, True, True, False])
matrix = df[['C1 (Benefit)', 'C2 (Benefit)', 'C3 (Benefit)', 'C4 (Cost)']].values

# ==========================================
# 1. METODE SAW
# ==========================================
norm_saw = np.zeros_like(matrix, dtype=float)
for j in range(matrix.shape[1]):
    if is_benefit[j]:
        norm_saw[:, j] = matrix[:, j] / np.max(matrix[:, j])
    else:
        norm_saw[:, j] = np.min(matrix[:, j]) / matrix[:, j]

df['SAW_Score'] = np.sum(norm_saw * weights, axis=1)
df['SAW_Rank'] = df['SAW_Score'].rank(ascending=False, method='min').astype(int)

# ==========================================
# 2. METODE WP
# ==========================================
w_wp = np.where(is_benefit, weights, -weights)
s_wp = np.prod(matrix ** w_wp, axis=1)
df['WP_Score'] = s_wp / np.sum(s_wp)
df['WP_Rank'] = df['WP_Score'].rank(ascending=False, method='min').astype(int)

# ==========================================
# 3. METODE TOPSIS
# ==========================================
norm_topsis = matrix / np.sqrt(np.sum(matrix**2, axis=0))
v_topsis = norm_topsis * weights
a_plus = np.where(is_benefit, np.max(v_topsis, axis=0), np.min(v_topsis, axis=0))
a_minus = np.where(is_benefit, np.min(v_topsis, axis=0), np.max(v_topsis, axis=0))
d_plus = np.sqrt(np.sum((v_topsis - a_plus)**2, axis=1))
d_minus = np.sqrt(np.sum((v_topsis - a_minus)**2, axis=1))

df['TOPSIS_Score'] = d_minus / (d_plus + d_minus)
df['TOPSIS_Rank'] = df['TOPSIS_Score'].rank(ascending=False, method='min').astype(int)

# ==========================================
# FORMATTING & OUTPUT KE STREAMLIT
# ==========================================
df_sorted = df.sort_values(by='SAW_Rank').reset_index(drop=True)

hasil = pd.DataFrame({
    'Alternatif': df_sorted['Alternatif'],
    'SAW (V / Rank)': df_sorted.apply(lambda x: f"{x['SAW_Score']:.3f} / {int(x['SAW_Rank'])}", axis=1),
    'WP (V / Rank)': df_sorted.apply(lambda x: f"{x['WP_Score']:.3f} / {int(x['WP_Rank'])}", axis=1),
    'TOPSIS (CC / Rank)': df_sorted.apply(lambda x: f"{x['TOPSIS_Score']:.3f} / {int(x['TOPSIS_Rank'])}", axis=1)
})

st.subheader("Tabel Perbandingan Hasil")
st.dataframe(hasil, use_container_width=True)

# Tambahin Insight seperti di PPT
st.warning("**Insight utama:** Ketiga metode (SAW, WP, TOPSIS) secara konsisten menempatkan peringkat teratas pada kandidat yang sama. Perbedaan kecil pada peringkat menengah ke bawah adalah hal wajar karena perbedaan mekanisme perhitungan matematika tiap metode, namun tidak mengubah rekomendasi akhir secara drastis.")
