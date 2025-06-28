from fastapi import APIRouter, HTTPException
from app.schemas.schemas import ChatTranscript, ChatInsights
from app.utils.loader import DataLoader
from app.utils.cleaner import DataCleaner
from app.utils.transformer import DataTransformer
from app.services.summarizer import ChatSummarizer
from app.core.logger import get_logger
import pandas as pd

router = APIRouter()
logger = get_logger()

# Load full dataset once
loader = DataLoader("data/BiztelAI_DS_Dataset_V1.json")
df = loader.load_data()
df_clean = DataCleaner(df).clean_all()
df_final = DataTransformer(df_clean).transform_all()
summarizer = ChatSummarizer()

@router.get("/summary")
async def get_summary():
    """Return dataset-level stats"""
    try:
        summary = {
            "total_conversations": df["conversation_id"].nunique(),
            "total_messages": len(df),
            "agents": df["agent"].nunique(),
            "unique_articles": df["article_url"].nunique(),
        }
        return summary
    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/transform")
async def transform_chat(chat: ChatTranscript):
    """Return cleaned and transformed new transcript"""
    try:
        df_chat = pd.DataFrame([msg.dict() for msg in chat.content])
        df_clean = DataCleaner(df_chat).clean_all()
        df_transformed = DataTransformer(df_clean).transform_all()
        return df_transformed.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Transform error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to transform chat")

@router.post("/insights", response_model=ChatInsights)
async def get_chat_insights(chat: ChatTranscript):
    """Analyze transcript and return LLM-based summary, sentiments, article link"""
    try:
        result = summarizer.analyze_transcript(
            chat=[msg.dict() for msg in chat.content],
            article_url=chat.article_url
        )
        return result
    except Exception as e:
        logger.error(f"Insights error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze chat")
