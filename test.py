from app.services.summarizer import ChatSummarizer
import json

def test_summary():
    with open("data/BiztelAI_DS_Dataset_V1.json", "r") as f:
        data = json.load(f)

    transcript = next(iter(data.values()))
    chat = transcript["content"]
    article_url = transcript.get("article_url")

    summarizer = ChatSummarizer()
    result = summarizer.analyze_transcript(chat, article_url=article_url)

    print("=== CHAT SUMMARY ===")
    for key, val in result.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    test_summary()
# This script tests the ChatSummarizer service by loading a sample dataset,
# extracting a chat transcript, and generating a summary with sentiment analysis.