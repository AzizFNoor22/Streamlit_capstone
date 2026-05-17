import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec

# --- App Config ---
st.set_page_config(page_title='Health & Nutrition Insights', layout='wide')

# --- Constants ---
ALPHA = 0.05
COLORS = {'a': '#3266ad', 'b': '#1D9E75', 'bad': '#D85A30'}

# --- Helper functions ---
def bmi_category(bmi: float) -> str:
    if bmi < 16:
        return 'Extremely Weak'
    if bmi < 18.5:
        return 'Weak'
    if bmi < 25:
        return 'Normal'
    if bmi < 30:
        return 'Overweight'
    if bmi < 40:
        return 'Obesity'
    return 'Extreme Obesity'


def normal_weight_range(height_cm: float) -> tuple[float, float]:
    height_m = height_cm / 100
    return 18.5 * height_m**2, 24.9 * height_m**2


def run_ab_test(group_a: pd.Series, group_b: pd.Series,
                name_a: str, name_b: str, metric: str) -> dict:
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError(
            f"Both groups must contain observations. Got n_a={len(group_a)}, n_b={len(group_b)}"
        )

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    pooled_std = np.sqrt(
        ((len(group_a) - 1) * group_a.std(ddof=1)**2 +
         (len(group_b) - 1) * group_b.std(ddof=1)**2) /
        (len(group_a) + len(group_b) - 2)
    )
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std

    se_diff = np.sqrt(group_a.var(ddof=1)/len(group_a) +
                      group_b.var(ddof=1)/len(group_b))
    ci_low = (group_a.mean() - group_b.mean()) - 1.96 * se_diff
    ci_high = (group_a.mean() - group_b.mean()) + 1.96 * se_diff

    return {
        'metric': metric,
        'name_a': name_a,
        'n_a': len(group_a),
        'mean_a': group_a.mean(),
        'std_a': group_a.std(ddof=1),
        'name_b': name_b,
        'n_b': len(group_b),
        'mean_b': group_b.mean(),
        'std_b': group_b.std(ddof=1),
        'mean_diff': group_a.mean() - group_b.mean(),
        'ci_95': (ci_low, ci_high),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < ALPHA,
    }


def plot_hist(ax, ga, gb, la, lb, title, xlabel, sig, color_b='b'):
    cb = COLORS[color_b]
    ax.hist(ga, bins=25, alpha=0.7, color=COLORS['a'], label=la, density=True)
    ax.hist(gb, bins=25, alpha=0.5, color=cb, label=lb, density=True)
    ax.axvline(ga.mean(), color=COLORS['a'], lw=2, ls='--', label=f'Mean {la} = {ga.mean():.1f}')
    ax.axvline(gb.mean(), color=cb, lw=2, ls='--', label=f'Mean {lb} = {gb.mean():.1f}')
    ax.set_title(title + ('\n✅ Signifikan' if sig else '\n❌ Tidak signifikan'),
                 fontsize=11, color='#1D9E75' if sig else COLORS['bad'])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)

# --- App Header ---
st.title('Health & Nutrition Insights Dashboard')
st.write('Jelajahi dataset dan hitung BMI Anda secara langsung. Aplikasi ini juga menyajikan analisis data kesehatan dan visualisasi interaktif.')

st.markdown('''
## Cara Menggunakan Aplikasi Ini

* Masukkan `Berat Badan`, `Tinggi Badan`, dan `Target Berat Badan` pada bagian **Perencana BMI Personal**.
* Filter dataset historis di sidebar berdasarkan **Jenis Kelamin** dan **Status BMI**.
* Buka expander untuk melihat data mentah, statistik, dan hasil uji A/B.
* Jalankan aplikasi lokal dengan `streamlit run main_streamlit.py`.
''')

# --- Load Data ---
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if 'BMI_Category_Index' not in df.columns or 'BMI_Status' not in df.columns:
        bmi_status_map = {
            0: 'Extremely Weak',
            1: 'Weak',
            2: 'Normal',
            3: 'Overweight',
            4: 'Obesity',
            5: 'Extreme Obesity',
        }
        bins = [-np.inf, 16, 18.5, 25, 30, 40, np.inf]
        labels = [0, 1, 2, 3, 4, 5]
        df['BMI_Category_Index'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=False)
        df['BMI_Status'] = df['BMI_Category_Index'].map(bmi_status_map)
    return df

try:
    df_eda = load_data('df_eda.csv')
    st.success('Dataset df_eda.csv berhasil dimuat!')
except FileNotFoundError:
    st.error("Error: 'df_eda.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# --- 1. Interactive BMI & Weight Planner ---
st.header('Perencana BMI & Berat Badan Personal')
st.write('Masukkan detail Anda untuk menghitung BMI saat ini dan mengevaluasi target berat badan.')

col1, col2, col3 = st.columns(3)
with col1:
    user_weight = st.number_input('Berat Badan Saat Ini (kg)', min_value=30.0, max_value=300.0, value=70.0, step=0.5)
with col2:
    user_height = st.number_input('Tinggi Badan (cm)', min_value=100.0, max_value=250.0, value=170.0, step=1.0)
with col3:
    target_weight = st.number_input('Target Berat Badan (kg)', min_value=30.0, max_value=300.0, value=65.0, step=0.5)

user_bmi = user_weight / ((user_height / 100) ** 2)
target_bmi = target_weight / ((user_height / 100) ** 2)
user_status = bmi_category(user_bmi)
target_status = bmi_category(target_bmi)
normal_min, normal_max = normal_weight_range(user_height)
weight_diff = target_weight - user_weight

st.subheader('Hasil Anda')
metrics_col1, metrics_col2 = st.columns(2)
with metrics_col1:
    st.metric(label='Kalkulasi BMI', value=f'{user_bmi:.2f}', delta=f'{weight_diff:+.1f} kg')
    if user_bmi < 18.5:
        st.warning('**Status: Kekurangan Berat Badan (Underweight)**')
    elif user_bmi < 25.0:
        st.success('**Status: Berat Badan Normal**')
    elif user_bmi < 30.0:
        st.warning('**Status: Kelebihan Berat Badan (Overweight)**')
    else:
        st.error('**Status: Obesitas**')

with metrics_col2:
    st.metric(label='Target Berat Badan', value=f'{target_weight:.1f} kg')
    st.markdown(f'**Target BMI:** {target_bmi:.1f} ({target_status})')
    st.markdown(f'**Normal range:** {normal_min:.1f}–{normal_max:.1f} kg untuk tinggi {user_height:.0f} cm')
    if normal_min <= target_weight <= normal_max:
        st.success('Target berat badan Anda berada dalam kisaran BMI normal.')
    else:
        st.info('Target berat badan Anda berada di luar kisaran BMI normal.')
    if weight_diff < 0:
        st.info(f'Untuk menurunkan {abs(weight_diff):.1f} kg, gunakan defisit kalori moderat dan olahraga teratur.')
    elif weight_diff > 0:
        st.info(f'Untuk menaikkan {weight_diff:.1f} kg, fokus pada surplus kalori sehat dan latihan beban.')
    else:
        st.success('Anda sudah berada di berat target!')

st.divider()

# --- 2. Explorasi Data ---
st.header('Eksplorasi Dataset')

with st.expander('Tampilkan Pratinjau Data Mentah'):
    st.dataframe(df_eda.head())
    st.write(f'**Dimensi Data Saat Ini:** {df_eda.shape[0]} baris, {df_eda.shape[1]} kolom')

with st.expander('Tampilkan Informasi & Statistik Dataset'):
    info_df = pd.DataFrame({
        'Tipe Data': df_eda.dtypes.astype(str),
        'Jumlah Non-Null': df_eda.notnull().sum(),
    })
    st.write('**Informasi Kolom:**')
    st.dataframe(info_df)
    st.write('**Statistik Deskriptif:**')
    st.dataframe(df_eda.describe())

# --- 3. Sidebar Filtering ---
st.sidebar.header('🔍 Filter Dataset')
filtered_df = df_eda.copy()

gender_cols = [c for c in df_eda.columns if c.startswith('Gender_')]
if gender_cols:
    gender_labels = [c.replace('Gender_', '') for c in gender_cols]
    selected_gender = st.sidebar.selectbox('Pilih Jenis Kelamin', ['Semua'] + gender_labels)
    if selected_gender != 'Semua':
        filtered_df = filtered_df[filtered_df[f'Gender_{selected_gender}'] == 1]

if 'BMI_Status' in filtered_df.columns:
    bmi_options = ['Semua'] + sorted(filtered_df['BMI_Status'].unique().tolist())
    selected_bmi_status = st.sidebar.selectbox('Pilih Status BMI', bmi_options)
    if selected_bmi_status != 'Semua':
        filtered_df = filtered_df[filtered_df['BMI_Status'] == selected_bmi_status]

st.subheader('Pratinjau Data yang Difilter')
st.dataframe(filtered_df.head())
st.write(f'Bentuk data yang difilter: {filtered_df.shape[0]} baris, {filtered_df.shape[1]} kolom')

# --- 4. Visualisasi ---
st.subheader('Visualisasi Dataset')
if not filtered_df.empty and 'BMI_Status' in filtered_df.columns:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.write('**Distribusi Status BMI**')
        fig, ax = plt.subplots(figsize=(8, 5))
        bmi_counts = filtered_df['BMI_Status'].value_counts().sort_values(ascending=False)
        ax.bar(bmi_counts.index, bmi_counts.values, color=sns.color_palette('viridis', len(bmi_counts)))
        ax.set_xlabel('Status BMI')
        ax.set_ylabel('Jumlah')
        ax.set_xticklabels(bmi_counts.index, rotation=45)
        st.pyplot(fig)

    with chart_col2:
        st.write('**Jumlah Berdasarkan Status BMI**')
        st.bar_chart(bmi_counts)
else:
    st.info('Tidak ada data yang tersedia untuk ditampilkan atau kolom BMI_Status tidak ditemukan.')

lifestyle_cols = ['Daily_Steps', 'Exercise_Frequency', 'Sleep_Hours']
if not filtered_df.empty and all(col in filtered_df.columns for col in lifestyle_cols + ['BMI_Status']):
    st.subheader('Heatmap Korelasi: Variabel Gaya Hidup vs Status BMI')
    bmi_mapping = {
        'Extremely Weak': 0, 'Weak': 1, 'Normal': 2,
        'Overweight': 3, 'Obesity': 4, 'Extreme Obesity': 5,
    }
    corr_df = filtered_df.copy()
    corr_df['BMI_Status_Num'] = corr_df['BMI_Status'].map(bmi_mapping)
    corr_matrix = corr_df[lifestyle_cols + ['BMI_Status_Num']].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Matriks Korelasi Variabel Gaya Hidup dan Status BMI')
    st.pyplot(fig_corr)
else:
    st.write('Tidak cukup data atau kolom yang diperlukan untuk heatmap korelasi.')

# --- 5. A/B Testing ---
st.header('A/B Testing Summary')
st.markdown('Analisis perbedaan antara grup tidur, kebiasaan merokok, dan status BMI terhadap nilai kesehatan.')

r1 = None
r2 = None
r3 = None
if 'Sleep_Hours' in df_eda.columns and 'Cholesterol_Level' in df_eda.columns:
    group_sleep_low = df_eda.loc[df_eda['Sleep_Hours'] < 6, 'Cholesterol_Level']
    group_sleep_high = df_eda.loc[df_eda['Sleep_Hours'] >= 6, 'Cholesterol_Level']
    r1 = run_ab_test(group_sleep_low, group_sleep_high,
                     name_a='tidur < 6 jam', name_b='tidur ≥ 6 jam', metric='Cholesterol_Level')

if 'Smoking_Habit_Yes' in df_eda.columns and 'Caloric_Intake' in df_eda.columns and 'Recommended_Calories' in df_eda.columns:
    df_eda['caloric_gap'] = df_eda['Caloric_Intake'] - df_eda['Recommended_Calories']
    group_smoker = df_eda.loc[df_eda['Smoking_Habit_Yes'] == 1.0, 'caloric_gap']
    group_nonsmoker = df_eda.loc[df_eda['Smoking_Habit_Yes'] == 0.0, 'caloric_gap']
    r2 = run_ab_test(group_smoker, group_nonsmoker,
                     name_a='perokok', name_b='non-perokok', metric='Caloric_Intake − Recommended_Calories')

if 'BMI_Status' in df_eda.columns and 'Blood_Sugar_Level' in df_eda.columns:
    group_obese = df_eda.loc[df_eda['BMI_Status'] == 'Obesity', 'Blood_Sugar_Level']
    group_normal = df_eda.loc[df_eda['BMI_Status'] == 'Normal', 'Blood_Sugar_Level']
    r3 = run_ab_test(group_obese, group_normal,
                     name_a='Obesity', name_b='Normal', metric='Blood_Sugar_Level')

if r1 is not None:
    with st.expander('UJI 1: Sleep Duration vs Cholesterol Level'):
        st.markdown(f"**Metric**: {r1['metric']}")
        st.markdown(f"**Group A ({r1['name_a']})**: n={r1['n_a']:,}, Mean={r1['mean_a']:.2f}, Std={r1['std_a']:.2f}")
        st.markdown(f"**Group B ({r1['name_b']})**: n={r1['n_b']:,}, Mean={r1['mean_b']:.2f}, Std={r1['std_b']:.2f}")
        st.markdown(f"**Mean Difference**: {r1['mean_diff']:+.2f} (95% CI: [{r1['ci_95'][0]:.2f}, {r1['ci_95'][1]:.2f}])")
        st.markdown(f"**P-value**: {r1['p_value']:.6f} {'✅ SIGNIFICANT' if r1['significant'] else '❌ NOT SIGNIFICANT'}")
        st.markdown(f"**Cohen's d**: {r1['cohens_d']:.4f}")
else:
    st.info('UJI 1 tidak dapat dijalankan karena kolom yang diperlukan tidak tersedia.')

if r2 is not None:
    with st.expander('UJI 2: Smoking Habit vs Caloric Gap'):
        st.markdown(f"**Metric**: {r2['metric']}")
        st.markdown(f"**Group A ({r2['name_a']})**: n={r2['n_a']:,}, Mean={r2['mean_a']:.2f}, Std={r2['std_a']:.2f}")
        st.markdown(f"**Group B ({r2['name_b']})**: n={r2['n_b']:,}, Mean={r2['mean_b']:.2f}, Std={r2['std_b']:.2f}")
        st.markdown(f"**Mean Difference**: {r2['mean_diff']:+.2f} (95% CI: [{r2['ci_95'][0]:.2f}, {r2['ci_95'][1]:.2f}])")
        st.markdown(f"**P-value**: {r2['p_value']:.6f} {'✅ SIGNIFICANT' if r2['significant'] else '❌ NOT SIGNIFICANT'}")
        st.markdown(f"**Cohen's d**: {r2['cohens_d']:.4f}")
else:
    st.info('UJI 2 tidak dapat dijalankan karena kolom yang diperlukan tidak tersedia.')

if r3 is not None:
    with st.expander('UJI 3: BMI Status vs Blood Sugar Level'):
        st.markdown(f"**Metric**: {r3['metric']}")
        st.markdown(f"**Group A ({r3['name_a']})**: n={r3['n_a']:,}, Mean={r3['mean_a']:.2f}, Std={r3['std_a']:.2f}")
        st.markdown(f"**Group B ({r3['name_b']})**: n={r3['n_b']:,}, Mean={r3['mean_b']:.2f}, Std={r3['std_b']:.2f}")
        st.markdown(f"**Mean Difference**: {r3['mean_diff']:+.2f} (95% CI: [{r3['ci_95'][0]:.2f}, {r3['ci_95'][1]:.2f}])")
        st.markdown(f"**P-value**: {r3['p_value']:.6f} {'✅ SIGNIFICANT' if r3['significant'] else '❌ NOT SIGNIFICANT'}")
        st.markdown(f"**Cohen's d**: {r3['cohens_d']:.4f}")
else:
    st.info('UJI 3 tidak dapat dijalankan karena kolom yang diperlukan tidak tersedia.')

if r1 is not None and r2 is not None and r3 is not None:
    st.subheader('A/B Testing Visualizations')
    fig_ab = plt.figure(figsize=(15, 12))
    fig_ab.suptitle('A/B Testing — Health & Nutrition Dataset', fontsize=14, fontweight='bold', y=0.98)
    gs_ab = gridspec.GridSpec(2, 3, figure=fig_ab, hspace=0.45, wspace=0.35)
    plot_hist(fig_ab.add_subplot(gs_ab[0, 0]),
              group_sleep_low, group_sleep_high,
              'Tidur < 6 jam', 'Tidur ≥ 6 jam',
              'Uji 1: Kolesterol vs Tidur', 'Cholesterol Level', r1['significant'])
    plot_hist(fig_ab.add_subplot(gs_ab[0, 1]),
              group_smoker, group_nonsmoker,
              'Perokok', 'Non-perokok',
              'Uji 2: Selisih Kalori vs Merokok', 'Caloric Gap (kcal)', r2['significant'],
              color_b='bad')
    plot_hist(fig_ab.add_subplot(gs_ab[0, 2]),
              group_obese, group_normal,
              'Obesitas', 'Normal',
              'Uji 3: Gula Darah vs BMI', 'Blood Sugar Level', r3['significant'],
              color_b='b')
    ax_sum = fig_ab.add_subplot(gs_ab[1, :])
    labels = [
        'Uji 1A\n(tidur<6)', 'Uji 1B\n(tidur≥6)',
        'Uji 2A\n(perokok)', 'Uji 2B\n(non-perokok)',
        'Uji 3A\n(obesitas)', 'Uji 3B\n(normal)',
    ]
    means = [r1['mean_a'], r1['mean_b'], r2['mean_a'], r2['mean_b'], r3['mean_a'], r3['mean_b']]
    errors = [1.96 * r1['std_a']/np.sqrt(r1['n_a']), 1.96 * r1['std_b']/np.sqrt(r1['n_b']),
              1.96 * r2['std_a']/np.sqrt(r2['n_a']), 1.96 * r2['std_b']/np.sqrt(r2['n_b']),
              1.96 * r3['std_a']/np.sqrt(r3['n_a']), 1.96 * r3['std_b']/np.sqrt(r3['n_b'])]
    bar_colors = [COLORS['a'], COLORS['b'], COLORS['a'], COLORS['bad'], COLORS['bad'], COLORS['b']]
    x = np.arange(len(labels))
    ax_sum.bar(x, means, color=bar_colors, alpha=0.8, yerr=errors, capsize=4, error_kw={'elinewidth': 1})
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(labels, fontsize=9)
    ax_sum.set_ylabel('Mean value', fontsize=10)
    ax_sum.set_title('Perbandingan Mean Antar Grup (error bars = 95% CI)', fontsize=11)
    ax_sum.tick_params(labelsize=8)
    ax_sum.grid(axis='y', alpha=0.3)
    for i, (r, pos) in enumerate([(r1, 0.5), (r2, 2.5), (r3, 4.5)]):
        sig_str = f"p={r['p_value']:.4f} {'✅' if r['significant'] else '❌'}"
        ax_sum.annotate(sig_str,
                        xy=(pos, max(means[i*2], means[i*2+1]) + max(errors[i*2], errors[i*2+1]) * 1.5),
                        ha='center', fontsize=9,
                        color='#1D9E75' if r['significant'] else COLORS['bad'])
    st.pyplot(fig_ab)
