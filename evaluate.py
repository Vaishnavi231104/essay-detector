import json
from app import analyze_essay

def run_evaluation():
    with open("data/dataset.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("=" * 75)
    print("VERITAS ADMISSIONS ESSAY DETECTOR — EMPIRICAL BENCHMARK")
    print("=" * 75)

    results = []

    for item in cases:
        report = analyze_essay(item["text"])
        score = report["overall_score"]
        verdict = report["overall_verdict"]
        ground_truth = item["ground_truth"]

        # Classification boundary: Score >= 50 is classified as AI/Mixed
        pred_is_ai = score >= 50
        actual_is_ai = ground_truth in ["ai", "mixed"]
        is_accurate = (pred_is_ai == actual_is_ai)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "ground_truth": ground_truth,
            "score": score,
            "verdict": verdict,
            "burstiness_cv": report["burstiness_cv"],
            "entropy": report["overall_entropy"],
            "ttr": report["lexical_diversity"],
            "markers": report["total_markers"],
            "esl_flag": report["esl_flag"],
            "is_accurate": is_accurate
        })

        status = "✅ PASS" if is_accurate else "❌ FAILURE (Confidently Wrong)"
        print(f"\n[{item['id']}] {item['category']} — {item['subject']}")
        print(f"  Ground Truth   : {ground_truth.upper()}")
        print(f"  Predicted Risk : {score}% ({verdict})")
        print(f"  Signals        : Burstiness CV={report['burstiness_cv']} | Entropy={report['overall_entropy']} | Markers={report['total_markers']}")
        print(f"  ESL Trigger    : {report['esl_flag']}")
        print(f"  Evaluation     : {status}")

    total = len(results)
    correct = sum(1 for r in results if r["is_accurate"])
    print("\n" + "=" * 75)
    print(f"SUMMARY: {correct}/{total} correctly classified ({round((correct/total)*100, 1)}%)")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()