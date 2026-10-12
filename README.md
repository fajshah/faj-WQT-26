# WTQ 2026 - The Utility Bill Decoder

An AI-powered utility bill extraction and Q&A system built using **Google GenAI SDK**, **Pydantic**, **Pandas**, and **Streamlit**.

---

## 🚀 Features & Pipeline

1. **Level 1 (Structured JSON Extraction):**
   - Extracts standardized utility data (Provider: KE, LESCO, IESCO, tariff, sanctioned load, dates, previous/current readings, units consumed, detailed charges breakdown, taxes breakdown, and billing amounts) strictly adhering to Pydantic schemas.
   - Outputs results to `output/level1.csv`.

2. **Level 2 (Bill Q&A & Explanations):**
   - Answers specific customer questions grounded directly in the line items and values extracted from the bill image.
   - Outputs results to `output/level2.csv`.

3. **Streamlit Interactive Web Dashboard (`app.py`):**
   - Visual browser UI to inspect extracted JSONs, view Level 2 Q&A results side-by-side with original bill images, and explore raw dataset tables.

---

## 🛠️ Setup & Installation

1. **Clone Repository:**
   ```bash
   git clone https://github.com/fajshah/faj-WQT-26.git
   cd faj-WQT-26
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r source/requirements.txt
   pip install streamlit
   ```

3. **Set API Key:**
   Set your Google Gemini API key as an environment variable:
   ```bash
   $env:GEMINI_API_KEY="your_api_key_here"
   ```

---

## 🏃 Execution

### 1. Run Extraction Pipeline
```bash
python main.py
```

### 2. Run Browser UI (Streamlit Dashboard)
```bash
streamlit run app.py
```

---

## 🤖 AI Usage Declaration
- **Model:** Gemini Flash-Lite (`gemini-3.1-flash-lite`)
- **Libraries:** Google GenAI SDK, Pandas, Pydantic, Streamlit, Pillow
