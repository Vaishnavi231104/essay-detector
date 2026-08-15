import streamlit as st
import numpy as np
import re
import math
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="Veritas | AI Admissions Essay Detector",
    page_icon="🔍",
    layout="wide",
)

# Stylometric & LLM Marker Constants
AI_MARKERS = {
    "delve", "delving", "testament", "tapestry", "beacon", "catalyst",
    "pivotal", "transformative", "crucible", "unwavering", "resonate",
    "furthermore", "moreover", "in conclusion", "it is worth noting",
    "plays a crucial role", "serves as a reminder", "embark", "foster",
    "intertwined", "multifaceted", "holistic", "paramount", "myriad",
    "undeniably", "harmonious", "cornerstone", "embarks"
}

def clean_words(text):
    return re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())

def calculate_shannon_entropy(words):
    if not words:
        return 0.0
    freqs = Counter(words)
    total = len(words)
    entropy = -sum((count / total) * math.log2(count / total) for count in freqs.values())
    return round(entropy, 3)

def calculate_burstiness(sentence_lengths):
    if len(sentence_lengths) < 2:
        return 0.0, 0.0
    mean_len = float(np.mean(sentence_lengths))
    std_dev = float(np.std(sentence_lengths))
    cv = std_dev / mean_len if mean_len > 0 else 0.0
    return round(mean_len, 2), round(cv, 3)

def analyze_essay(text):
    if not text.strip():
        return None

    # Sentence Tokenization
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text]

    all_words = clean_words(text)
    total_words = len(all_words)
    total_sentences = len(raw_sentences)

    sentence_word_counts = [len(clean_words(s)) for s in raw_sentences]
    avg_len, burstiness_cv = calculate_burstiness(sentence_word_counts)
    
    unique_words = set(all_words)
    lexical_diversity = round(len(unique_words) / total_words, 3) if total_words > 0 else 0.0
    overall_entropy = calculate_shannon_entropy(all_words)

    # Sentence-level analysis
    sentence_records = []
    total_weighted_ai_score = 0.0
    total_marker_count = 0

    for idx, s in enumerate(raw_sentences):
        words = clean_words(s)
        w_count = len(words)
        w_set = set(words)
        ttr = round(len(w_set) / w_count, 3) if w_count > 0 else 0.0
        entropy = calculate_shannon_entropy(words)

        # Feature 1: Marker occurrences
        markers_found = [w for w in words if w in AI_MARKERS]
        total_marker_count += len(markers_found)
        marker_signal = min(len(markers_found) * 0.35, 0.6)

        # Feature 2: Uniformity of sentence length
        len_dev = abs(w_count - avg_len)
        uniformity_signal = 0.35 if len_dev < 3 and w_count > 10 else 0.05

        # Feature 3: Lexical smoothness
        smoothness_signal = 0.25 if (entropy > 2.8 and ttr > 0.85 and w_count > 8) else 0.05

        raw_prob = marker_signal + uniformity_signal + smoothness_signal
        if w_count < 6:
            raw_prob *= 0.4

        ai_prob = min(max(round(raw_prob, 2), 0.02), 0.98)

        reasons = []
        if markers_found:
            reasons.append(f"Contains LLM-favored marker(s): {', '.join(markers_found)}")
        if uniformity_signal > 0.2:
            reasons.append("Syntactic uniformity matching synthetic pacing")
        if smoothness_signal > 0.2:
            reasons.append("Uncharacteristically high lexical smoothness")
        if not reasons:
            reasons.append("Natural burstiness and organic phrasing")

        if ai_prob >= 0.60:
            verdict = "Likely Machine Generated"
            color = "#fee2e2"
            border = "#ef4444"
        elif ai_prob >= 0.35:
            verdict = "Mixed / Polished"
            color = "#fef3c7"
            border = "#f59e0b"
        else:
            verdict = "Likely Human"
            color = "#dcfce7"
            border = "#22c55e"

        total_weighted_ai_score += ai_prob * max(w_count, 1)

        sentence_records.append({
            "index": idx + 1,
            "text": s,
            "word_count": w_count,
            "entropy": entropy,
            "ttr": ttr,
            "ai_prob": int(ai_prob * 100),
            "verdict": verdict,
            "reasons": reasons,
            "bg_color": color,
            "border_color": border
        })

    # Overall Essay Score
    base_score = (total_weighted_ai_score / max(total_words, 1)) * 100
    if burstiness_cv < 0.28:
        base_score += 15
    elif burstiness_cv > 0.50:
        base_score -= 15

    overall_score = int(min(max(round(base_score), 0), 100))

    if overall_score >= 65:
        overall_verdict = "Likely Machine Generated"
    elif overall_score >= 35:
        overall_verdict = "Mixed / Partially Polished"
    else:
        overall_verdict = "Likely Human Written"

    # ESL False Positive Heuristic
    esl_flag = burstiness_cv < 0.35 and total_marker_count == 0 and lexical_diversity < 0.45

    return {
        "overall_score": overall_score,
        "overall_verdict": overall_verdict,
        "total_words": total_words,
        "total_sentences": total_sentences,
        "avg_len": avg_len,
        "burstiness_cv": burstiness_cv,
        "overall_entropy": overall_entropy,
        "lexical_diversity": lexical_diversity,
        "total_markers": total_marker_count,
        "sentences": sentence_records,
        "esl_flag": esl_flag
    }

# UI Layout
st.title("🔍 Veritas — Explainable Admissions Essay Detector")
st.caption("A statistical stylometric detector using sentence burstiness variance, Shannon entropy, and formulaic marker density.")

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Admissions Essay Input")
    sample_text = st.text_area(
        "Paste the candidate's essay here:",
        height=320,
        placeholder="Paste college admissions essay here..."
    )
    analyze_btn = st.button("Analyze Essay Signals", type="primary", use_container_width=True)

if analyze_btn and sample_text:
    report = analyze_essay(sample_text)

    with col2:
        st.subheader("Aggregate Assessment")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("AI Probability", f"{report['overall_score']}%")
        m_col2.metric("Verdict", report['overall_verdict'])

        st.markdown("### Evidence Signals")
        sig_col1, sig_col2, sig_col3 = st.columns(3)
        sig_col1.metric("Burstiness (CV)", f"{report['burstiness_cv']}")
        sig_col2.metric("Shannon Entropy", f"{report['overall_entropy']}")
        sig_col3.metric("Type-Token Ratio", f"{report['lexical_diversity']}")

        if report['esl_flag']:
            st.warning("⚠️ **ESL Risk Indicator:** Detected low lexical variance without synthetic marker vocabulary. This pattern frequently mirrors Non-Native English writing rather than AI generation.")

    st.markdown("---")
    st.subheader("Sentence-by-Sentence Breakdown & Evidence")

    # Render colored text segments
    html_spans = []
    for s in report['sentences']:
        html_spans.append(
            f"<span style='background-color: {s['bg_color']}; border-bottom: 2px solid {s['border_color']}; padding: 2px 6px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 6px;' title='AI Prob: {s['ai_prob']}% | {s['reasons'][0]}'>{s['text']}</span>"
        )
    st.markdown("".join(html_spans), unsafe_allow_html=True)

    st.markdown("#### Individual Sentence Evidence Log")
    for s in report['sentences']:
        with st.expander(f"Sentence {s['index']}: {s['verdict']} ({s['ai_prob']}% AI Risk) — \"{s['text'][:60]}...\""):
            st.write(f"**Full Text:** {s['text']}")
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Word Count:** {s['word_count']}")
            c2.write(f"**Entropy:** {s['entropy']}")
            c3.write(f"**Lexical Diversity (TTR):** {s['ttr']}")
            st.write("**Identified Drivers:**")
            for r in s['reasons']:
                st.write(f"- {r}")