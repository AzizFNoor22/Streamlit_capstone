# Health & Nutrition Insights (Streamlit)

An interactive Streamlit dashboard for exploring patterns in the **Personalized Medical Diet Recommendations** dataset and running simple A/B (hypothesis) tests on selected health metrics.

## Features
- Data overview (preview of `df_eda.csv`)
- BMI status distribution visualization
- A/B testing summary (re-run at runtime)
- A/B testing visualizations (histograms + mean comparison with 95% CI error bars)

## Project Files
- `main_streamlit.py` — Streamlit app entrypoint
- `df_eda.csv` — dataset used for visualization + A/B tests
- `df_clean.csv` — cleaned dataset (not directly used by the current app, but kept in repo)

## A/B Tests Implemented
The app performs these statistical comparisons (two-sample t-test, Welch style `equal_var=False`):
1. **Sleep duration**: `Sleep_Hours < 6` vs `Sleep_Hours >= 6` → `Cholesterol_Level`
2. **Smoking habit**: `Smoking_Habit_Yes == 1` vs `Smoking_Habit_Yes == 0` → (Caloric gap) 
   - `caloric_gap = Caloric_Intake - Recommended_Calories`
3. **BMI status**: `BMI_Status == 'Obesity'` vs `BMI_Status == 'Normal'` → `Blood_Sugar_Level`

For each test, the app reports:
- sample sizes (`n_a`, `n_b`)
- means/std devs
- mean difference
- 95% CI for the mean difference (approx. using SE)
- t-statistic and p-value
- Cohen’s d effect size
- significance at `alpha = 0.05`

## How to Run
### 1) Create/activate an environment (recommended)
```bash
python -m venv .venv
.
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scipy matplotlib seaborn
```

### 3) Start the app
```bash
python -m streamlit run main_streamlit.py --server.port 8501
```
Then open: `http://localhost:8501`

## Notes / Troubleshooting
- The app uses relative paths: `load_data('df_eda.csv')`. Ensure you run the command from the project root (`d:/streamlit_capstone`).
- If emojis (✅ / ❌) render as missing-glyph warnings, this is a font configuration issue only (not a data or statistical error).

## License
Add a license of your choice (e.g., MIT). 

