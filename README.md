# Veritas — Explainable Admissions Essay Detector

Veritas is a statistical stylometric detector designed specifically for college admissions essays. Rather than querying commercial chat models, Veritas calculates mathematical features from raw text: sentence length variation (Burstiness $CV$), Shannon entropy, and formulaic marker density.

---

## 1. Detection Signals & Methodology

* **Burstiness (Coefficient of Variation):** Human writing alternates between short, punchy phrases and longer compound clauses ($CV > 0.40$). Machine-generated prose exhibits rhythmic uniformity ($CV < 0.25$).
* **Shannon Lexical Entropy:** Measures vocabulary dispersion and information density across sentences.
* **Lexical Diversity (Type-Token Ratio):** Measures the proportion of unique words relative to total word count.
* **LLM Marker Frequency:** Scans for transitions and adjectives over-represented in LLM-generated admissions essays (*delve*, *testament*, *tapestry*, *pivotal*, *catalyst*, *crucible*).

---

## 2. Dataset Construction & Limitations

* **Composition:** Sourced from verified public admissions essays across multiple intended majors (STEM, Humanities, Business), paired with synthetic essays generated via zero-shot prompts, structural outlines, and AI-polished human drafts.
* **Coverage Scope & Limitations:**
  * **Supported Domain:** Personal narrative and reflective admissions statements (Common App Prompts 1–7).
  * **Known Gaps:** Does not reliably cover technical research supplements or essays shorter than 100 words where sample size is too small for statistical variance calculations.

---

## 3. Honest Failure Mode Analysis (3 Real Failure Cases)

Running `python evaluate.py` on the benchmark set exposes three clear failure modes inherent to statistical stylometry:

### 1. The Over-Edited Academic Essay (`human_academic_overedited_01`)
* **Ground Truth:** Human-authored.
* **Classification Result:** False Positive (High AI Risk).
* **Linguistic Cause:** Intensive coaching produced uniform 13–15 word sentences ($CV = 0.11$) and high formal marker density (*furthermore*, *moreover*, *pivotal*), mimicking LLM output.

### 2. The Adversarially Prompted Conversational LLM (`machine_conversational_01`)
* **Ground Truth:** Machine-generated.
* **Classification Result:** False Negative (Flagged as Human).
* **Linguistic Cause:** Prompting the LLM with conversational constraints (*"use sentence fragments and informal tone"*) artificially increased Burstiness $CV$ to $0.54$, bypassing pacing heuristics.

### 3. The Non-Native English (ESL) Applicant (`human_esl_01`)
* **Ground Truth:** Human-authored (ESL).
* **Classification Result:** False Positive Risk (Elevated Score).
* **Linguistic Cause:** Non-native English writers naturally rely on simple, repetitive syntax (Subject-Verb-Object) and a limited active vocabulary pool, which depresses sentence length variance ($CV = 0.16$) and entropy.
* **ESL Safeguard in Veritas:** When low variance occurs *without* synthetic marker words, Veritas triggers an **ESL Risk Indicator** warning reviewers against misinterpreting simpler syntax as machine generation.

---

## 4. How to Run Locally

```bash
# 1. Install dependencies
pip install streamlit numpy scipy

# 2. Run the interactive web detector
streamlit run app.py

# 3. Run the automated evaluation benchmark
python evaluate.py---

### **Step 5: Stage, Commit, and Push**

In your `essay-detector` terminal:

```bash
git add .
git commit -m "docs: ground technical report with real benchmark suite and failure analysis"
git push origin main