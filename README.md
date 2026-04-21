
# AI Career Match 🚀

## 📌 Overview

AI Career Match is an intelligent system that analyzes your resume and matches it with real-world job listings. It identifies skill gaps and provides a personalized learning roadmap to help improve career readiness.

---

## 💡 Features

* 📄 Resume parsing (PDF, DOCX, TXT)
* 🧠 AI-based job matching using TF-IDF & Cosine Similarity
* 🔍 Skill extraction from resume & job descriptions
* 📊 Match score calculation
* 📉 Skill gap analysis
* 🗺️ Personalized learning roadmap
* 🌐 Live job fetching using Adzuna API

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Scikit-learn
* NLTK
* Requests

---

## 📂 Project Structure

* `ai_career_match.py` → Main application
* `requirements.txt` → Dependencies

---

## ▶️ How to Run (Step-by-Step)

### 🔹 Step 1: Clone the repository

```bash id="a1x92k"
git clone https://github.com/YOUR-USERNAME/AI-Career-Match.git
cd AI-Career-Match
```

---

### 🔹 Step 2: Install dependencies

```bash id="b7m3qz"
pip install -r requirements.txt
```

---

### 🔹 Step 3: Set up Adzuna API 🔑

1. Go to 👉 https://developer.adzuna.com
2. Sign up / Create an account
3. Create an app to get:

   * `APP ID`
   * `APP KEY`

---

### 🔹 Step 4: Add API keys (IMPORTANT)

Open `ai_career_match.py` and replace:

```python id="g8xk2v"
ADZUNA_APP_ID = "YOUR_APP_ID"
ADZUNA_APP_KEY = "YOUR_APP_KEY"
```

👉 Paste your actual keys here

⚠️ Do NOT upload real API keys publicly (use `.env` in production)

---

### 🔹 Step 5: Run the application

```bash id="l3n8zp"
streamlit run ai_career_match.py
```

---

## 🧑‍💻 How to Use (User Guide)

### 1️⃣ Upload Resume

* Upload PDF / DOCX / TXT
  OR
* Paste resume text manually

---

### 2️⃣ Select Job Roles

Choose roles like:

* Data Scientist
* Data Analyst
* Machine Learning Engineer

---

### 3️⃣ Click Analyze

Click **“Analyze & Match Jobs”**

---

### 4️⃣ View Results

You will see:

✅ Match Score (0–100%)
✅ Best matching jobs
✅ Skills you already have
✅ Missing skills (gap analysis)

---

### 5️⃣ Explore Tabs

* **Dashboard** → Summary
* **Jobs** → All job matches
* **Analyze** → Detailed skill comparison
* **Roadmap** → Learning path

---

## 🔐 Security Note

* Do NOT expose API keys in GitHub
* Use environment variables (`.env`) in real projects

---

## 📈 Future Improvements

* Better NLP-based skill extraction
* More job APIs integration
* UI/UX improvements
* Deployment (Streamlit Cloud)

---

✨ Created by Namandeep Kaur
