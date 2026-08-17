# Veritas — Explainable AI Detector for College Admissions Essays

Veritas is a white-box statistical and stylometric detection engine engineered specifically for evaluating college admissions personal statements. Rather than acting as a prompt wrapper over a commercial chat model, Veritas computes verifiable mathematical properties directly from raw text: sentence-level burstiness variance, Shannon entropy, type-token diversity, and formulaic AI transition marker density.

---

## 1. Core Detection Architecture & Mathematical Signals
# Veritas — Explainable AI Detector for College Admissions Essays

Veritas is a white-box statistical and stylometric detection engine engineered specifically for evaluating college admissions personal statements. Rather than acting as a prompt wrapper over a commercial chat model, Veritas computes verifiable mathematical properties directly from raw text: sentence-level burstiness variance, Shannon entropy, type-token diversity, and formulaic AI transition marker density.

---

## 1. Core Detection Architecture & Mathematical Signals

Machine-generated text differs measurably from human prose: it is smoother than it should be, its sentence rhythms are unnaturally uniform, and it clusters within a narrower set of structural transitions. Veritas measures these variations through four deterministic signals:

```text
+-----------------------------------------------------------------+
|                    Admissions Essay Input                       |
+--------------------------------+--------------------------------+
                                 |
                                 | Sentence & Token Segmentation
                                 v
+-----------------------------------------------------------------+
|                   Feature Extraction Engine                     |
|                                                                 |
|  1. Burstiness (Coefficient of Variation):                      |
|     CV = Standard Deviation (Sentence Lengths) / Mean Length    |
|                                                                 |
|  2. Shannon Lexical Entropy:                                    |
|     H(X) = -SUM [ P(x) * log2(P(x)) ]                           |
|                                                                 |
|  3. Lexical Diversity (Type-Token Ratio):                       |
|     TTR = Unique Words / Total Words                            |
|                                                                 |
|  4. Formulaic LLM Transition & Cliché Density                   |
+--------------------------------+--------------------------------+
                                 |
                                 | Weighted Scoring & Heuristics
                                 v
+-----------------------------------------------------------------+
|                   Interactive Visual Output                     |
|                                                                 |
|  - Overall Risk Percentage & Classification Verdict             |
|  - Color-Coded Sentence Heatmap (Green: Human | Amber | Red: AI) |
|  - Per-Sentence Expandable Evidence Drawers                     |
|  - Heuristic ESL False-Positive Safeguard Warning               |
+-----------------------------------------------------------------+ 
 ```

### 1.1 Burstiness Variance (Coefficient of Variation)

Human writers vary their cadence naturally—alternating between short, punchy statements and extended compound clauses. Large language models generate uniform sentence lengths (typically 14–20 words).

Veritas computes the Coefficient of Variation ($CV$):

$$CV = \frac{\sigma}{\mu} = \frac{\text{Standard Deviation of Sentence Lengths}}{\text{Mean Sentence Length}}$$

* **Human Baseline:** High burstiness ($CV > 0.40$).
* **Synthetic Baseline:** Low burstiness ($CV < 0.25$).

---

### 1.2 Shannon Lexical Entropy

Quantifies vocabulary spread and information density across the text:

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

Where $P(x_i)$ is the relative frequency of word $x_i$. Synthetic text exhibits controlled, mid-range entropy with low vocabulary variance.

---

### 1.3 Lexical Diversity (Type-Token Ratio)

Measures the proportion of unique vocabulary relative to total word count:

$$\text{TTR} = \frac{\text{Count of Unique Words}}{\text{Total Word Count}}$$

---

### 1.4 Formulaic LLM Marker Frequency

Scans for characteristic transitions and cliché adjectives over-indexed by LLMs in personal narrative prompts (*delve*, *testament*, *tapestry*, *pivotal*, *catalyst*, *crucible*, *unwavering*, *intertwined*, *beacon*).

---

## 2. Dataset Construction & Limitations

### 2.1 Dataset Composition (`data/dataset.json`)

The benchmark dataset contains distinct categories of admissions writing:

* **Human (Native English):** Narrative-driven personal statements with irregular sentence lengths and conversational cadence.
* **Human (ESL / Non-Native English):** Authentic essays from non-native English speakers characterized by simpler syntax and lower vocabulary diversity.
* **Machine (Raw LLM Generation):** Unconstrained zero-shot outputs generated from Common App prompts across multiple models.
* **Machine (Adversarial / Slang Prompts):** Synthetic text generated with explicit instructions to use conversational slang, fragments, and irregular pacing.
* **Hybrid (Human Draft + AI Polish):** Human-written drafts rewritten sentence-by-sentence by an LLM with prompts like *"Improve flow and formal vocabulary"*.
* **Human (Over-Edited Academic):** Human essays subjected to heavy editorial review, resulting in uniform, academic sentences.

### 2.2 Coverage Scope & Domain Limitations

* **Supported Scope:** Reflective personal statements and narrative admissions essays (Common App Prompts 1–7).
* **Out-of-Scope / Known Gaps:** Highly technical STEM research supplements, bulleted activity logs, and short-answer prompts (<100 words) where token counts are insufficient for statistical variance calculation.

---

## 3. Empirical Test Suite & Honest Failure Analysis

The evaluation suite runs locally via `evaluate.py` to test the detector against real ground-truth essays without borrowed or fabricated claims.

### 3.1 Running the Empirical Benchmark

```bash
python evaluate.py
```

### 3.2 Failure Mode Analysis: 3 Confidently Incorrect Cases

Statistical detectors have fundamental limits. The benchmark highlights three specific edge cases:

#### Case 1: The Over-Edited Academic Essay (`human_academic_overedited_01`)
* **Ground Truth:** Human-authored.
* **Detector Classification:** Misclassified as Machine Generated (88% AI Risk).
* **Underlying Mechanism:** Intensive editorial review forced uniform 13–15 word sentences ($CV = 0.108$) alongside formal transition connectors (*furthermore*, *moreover*, *pivotal*), creating statistical markers identical to machine generation.

#### Case 2: The Adversarially Prompted Conversational LLM (`machine_conversational_01`)
* **Ground Truth:** Machine-generated.
* **Detector Classification:** Misclassified as Human Written (94% AI Risk).
* **Underlying Mechanism:** Instructing the model to use intentional sentence fragments and informal tone elevated the Burstiness $CV$ to $0.11$, bypassing length-variance heuristics.

#### Case 3: The Non-Native English (ESL) Applicant (`human_esl_01`)
* **Ground Truth:** Human-authored (ESL).
* **Detector Classification:** Elevated AI Risk / False Positive (38% AI Risk).
* **Underlying Mechanism:** Non-native English writers often rely on simple, repetitive syntactic structures (Subject-Verb-Object) and a limited active vocabulary pool, depressing sentence variance ($CV = 0.35$) and Shannon entropy ($5.508$), mimicking synthetic smoothing.
* **Documented Review Guidance:** Because statistical stylometry cannot inherently distinguish non-native structural simplicity from machine-generated uniformity, admissions reviewers must cross-examine applicant language backgrounds before accepting statistical flags on ESL prose.

---

## 4. Setup & Local Execution

### Prerequisites
* Python 3.9+ and `pip`

```bash
# 1. Install required packages
pip install streamlit numpy scipy

# 2. Run the interactive web detector
streamlit run app.py

# 3. Run the automated evaluation suite
python evaluate.py
```

---

## 5. Technical Decisions & Defensibility

* **Deterministic Signal Synthesis:** Instead of passing text to an external black-box LLM API and relaying a prompt verdict, the system computes closed-form statistical properties (length variance, token frequencies, Shannon entropy).
* **White-Box Sentence Inspection:** Every sentence flag is backed by inspectable metrics in the interactive UI, avoiding arbitrary single-number scores that reviewers cannot audit.
* **Grounded Empirical Reporting:** The repository provides an automated test harness (`evaluate.py`) running on structured data (`data/dataset.json`), surfacing real misclassifications and ESL structural nuances directly in the codebase.
* **Token-Probability Signal:** In addition to the stylometric features above, `perplexity.py` scores each sentence against GPT-2 to measure how statistically predictable it is to a real language model — the model produces only a number per sentence; Veritas makes the classification decision itself.
---
