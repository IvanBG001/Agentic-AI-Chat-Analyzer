import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
from app.services.summarizer import ChatSummarizer
from app.services.metrics import EvaluationMetrics

def batch_process(filepath="data/BiztelAI_DS_Dataset_V1.json", output_csv="data/batch_summary_results.csv"):
    with open(filepath, 'r') as f:
        data = json.load(f)

    summarizer = ChatSummarizer()
    summaries = []
    sentiment_preds_1 = []
    sentiment_preds_2 = []
    sentiment_actual_1 = []
    sentiment_actual_2 = []
    ref_summaries = []  # Add manual reference if available

    for convo_id, convo in list(data.items())[:30]:  # Process first 30 for example
        chat = convo["content"]
        article_url = convo.get("article_url")

        result = summarizer.analyze_transcript(chat, article_url)
        summaries.append({
            "conversation_id": convo_id,
            **result
        })

        # For evaluation if ground truth exists
        for msg in chat:
            if msg["agent"] == "agent_1":
                sentiment_actual_1.append(msg["sentiment"])
            elif msg["agent"] == "agent_2":
                sentiment_actual_2.append(msg["sentiment"])

        sentiment_preds_1.append(result["agent_1_sentiment"])
        sentiment_preds_2.append(result["agent_2_sentiment"])

        # If you have human-generated reference summaries:
        ref_summaries.append("Insert gold summary here.")

    df = pd.DataFrame(summaries)
    df.to_csv(output_csv, index=False)
    print(f"✅ Batch results saved to {output_csv}")

    # Sentiment accuracy (optional if gold exists)
    if sentiment_actual_1:
        acc1 = EvaluationMetrics.evaluate_sentiment_accuracy(sentiment_preds_1, sentiment_actual_1[:len(sentiment_preds_1)])
        acc2 = EvaluationMetrics.evaluate_sentiment_accuracy(sentiment_preds_2, sentiment_actual_2[:len(sentiment_preds_2)])
        print(f"Agent 1 Sentiment Accuracy: {acc1:.2f}")
        print(f"Agent 2 Sentiment Accuracy: {acc2:.2f}")

    # BLEU Score (if you added real gold summaries)
    if "Insert gold summary here." not in ref_summaries:
        bleu_score = EvaluationMetrics.batch_bleu(df["summary"].tolist(), ref_summaries)
        print(f"Average BLEU Score: {bleu_score:.2f}")

if __name__ == "__main__":
    batch_process()
