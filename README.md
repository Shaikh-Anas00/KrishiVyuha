# 🌾 KrishiVyuha

> **Smart Crop Portfolio Advisor for Indian Farmers**  
> Tell us your land, location, and season — we tell you *which crops to plant and in what mix* to maximise profit while protecting against bad weather years.

---

## What it does

KrishiVyuha is a data-science-powered web app that helps farmers make smarter planting decisions.

1. **Predicts crop yield** — using 20+ years of historical crop + weather data from across India
2. **Optimises your crop mix** — like a financial portfolio, but for your farm land
3. **Accounts for real risks** — monsoon failures, price swings, El Niño/La Niña climate cycles
4. **Suggests a 3-season rotation plan** — so your soil stays healthy year after year
5. **Speaks plain language** — no λ values or math shown to the farmer; just "Aggressive / Balanced / Defensive" strategy cards

---

## Features

- 🗺️ **Location-aware** — enter your district and season, climate data is auto-fetched
- 🎙️ **Voice input** — speak your farm setup instead of typing
- 📊 **Visual allocation chart** — see the crop mix as an interactive donut chart
- 🛡️ **Survival budget check** — warns if your worst-case income falls below your minimum need
- ♻️ **Rotation optimizer** — 3-season plan with nitrogen-fixation bonuses for legumes
- 📋 **Full financial breakdown** — expected revenue, volatility, and PMFBY insurance payout per crop
- 🌐 **Live ENSO sync** — pulls the latest El Niño / La Niña index automatically

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 + React 19 + TailwindCSS + Recharts |
| Backend API | FastAPI (Python) |
| ML Model | Multiple Linear Regression (scikit-learn) |
| Optimiser | Markowitz Mean-Variance QP via SciPy SLSQP |
| Data | Pandas, NumPy |
| Runtime | Uvicorn (ASGI) |

---

## Data Engineering

- **345,000+ crop records** from India (1997–2024), merged with NOAA ONI climate index
- Hierarchical median imputation — fills missing values at district → state → crop level (no mean imputation shortcuts)
- Coconut yield unit correction (thousand nuts/ha vs tonnes/ha — different units in the raw government data)
- Outlier audit — kg/ha mis-entries converted to t/ha; extreme yields capped at crop-specific 99.5th percentile
- 816 one-hot encoded features (State × District × Crop × Season × Climate regime)
- Three validation schemes: random 80/20 split, temporal out-of-time split (train ≤2018, test ≥2019), and 5-fold cross-validation

### Model Performance

| Scheme | R² | MAE |
|---|---|---|
| Random split | 0.790 | 57.7 t/ha |
| Temporal split (future seasons) | **0.801** | 56.9 t/ha |
| 5-Fold CV | 0.789 ± 0.007 | 58.5 ± 1.8 t/ha |
| Field crops only (excl. Coconut) | — | **28.7 t/ha** |

---

## How to Run

**Backend** (Terminal 1, from project root):
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (Terminal 2):
```bash
cd agrivest-frontend
npm install
npm run dev
```

Open → **http://localhost:3000**  
API docs → **http://127.0.0.1:8000/docs**

---

## Project Structure

```
KrishiVyuha/
├── app/                  ← FastAPI backend
├── agrivest-frontend/    ← Next.js web app
├── src/                  ← Data science pipeline
│   ├── data_preprocessing.py
│   ├── eda_and_engineering.py
│   ├── yield_predictor.py
│   ├── portfolio_optimizer.py
│   └── rotation_optimizer.py
├── data/                 ← Raw + processed datasets
├── models/               ← Trained model + risk profiles
├── notebooks/            ← Jupyter demos (data science foundations)
└── reports/              ← EDA plots + project report
```

---

## Limitations (honest ones)

- Revenue risk uses price volatility estimated from mandi statistics — not live real-time prices
- The yield model doesn't account for soil quality or farm management practices
- Rotation optimizer is single-state (doesn't model cascading multi-year soil dynamics)
- Some horticultural crops use proxied covariance (e.g., Tomato ≈ Onion risk profile)

---

## Built for

Semester 5 Data Science Course Project — demonstrating end-to-end ML pipeline, portfolio optimisation, and farmer-first UI design.
