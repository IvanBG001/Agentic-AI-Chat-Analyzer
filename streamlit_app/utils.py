from app.services.summarizer import ChatSummarizer

summarizer = ChatSummarizer()

def run_analysis(chat, article_url=None):
    return summarizer.analyze_transcript(chat=chat, article_url=article_url)
