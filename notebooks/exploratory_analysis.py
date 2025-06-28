import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now the app module can be imported
from app.utils.loader import DataLoader
from app.utils.cleaner import DataCleaner
from app.utils.transformer import DataTransformer
from ydata_profiling import ProfileReport

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

def prepare_data():
    loader = DataLoader("data/BiztelAI_DS_Dataset_V1.json")
    df = loader.load_data()

    cleaner = DataCleaner(df)
    df_clean = cleaner.clean_all()

    transformer = DataTransformer(df_clean)
    df_final = transformer.transform_all()
    return df_final

def agent_message_stats(df):
    print("\n🔹 Agent-wise Message Count:")
    print(df["agent"].value_counts())
    
    sns.countplot(data=df, x="agent", palette="Set2")
    plt.title("Agent-wise Message Distribution")
    plt.xlabel("Agent")
    plt.ylabel("Message Count")
    plt.tight_layout()
    plt.show()

def sentiment_analysis(df):
    sentiment_summary = df.groupby("agent")["sentiment"].value_counts().unstack().fillna(0)
    print("\n🔹 Sentiment Count by Agent:")
    print(sentiment_summary)

    sentiment_summary.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="viridis")
    plt.title("Sentiment Distribution by Agent")
    plt.ylabel("Count")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.show()

def article_stats(df):
    article_summary = df.groupby("article_url")["message"].count().sort_values(ascending=False)
    print("\n🔹 Message Count by Article:")
    print(article_summary.head())

    plt.figure(figsize=(10, 5))
    sns.histplot(article_summary, bins=20, kde=True)
    plt.title("Article-wise Message Count Distribution")
    plt.xlabel("Messages per Article")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def wordcloud_agent(df, agent_name):
    text = " ".join(df[df["agent"] == agent_name]["processed_message"])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {agent_name}")
    plt.tight_layout()
    plt.show()

def heatmap_turn_ratings(df):
    pivot_table = pd.crosstab(df["agent"], df["turn_rating"])
    sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Turn Ratings Heatmap per Agent")
    plt.tight_layout()
    plt.show()

def summarize_transcripts(df):
    summary_df = df.groupby("conversation_id").agg(
        article_url=("article_url", "first"),
        agent_1_msgs=("agent", lambda x: (x == "agent_1").sum()),
        agent_2_msgs=("agent", lambda x: (x == "agent_2").sum()),
        agent_1_sentiment=("sentiment", lambda x: ", ".join(x[df["agent"] == "agent_1"].unique())),
        agent_2_sentiment=("sentiment", lambda x: ", ".join(x[df["agent"] == "agent_2"].unique())),
        total_messages=("message", "count")
    ).reset_index()

    print("\n🔹 Sample Transcript Summary:")
    print(summary_df.head())
    return summary_df

def generate_profiling_report(df):
    from ydata_profiling import ProfileReport  # or `from pandas_profiling import ProfileReport` if older version
    profile = ProfileReport(df, title="BiztelAI Chat Dataset Profile", explorative=True)
    profile.to_file("notebooks/BiztelAI_Chat_Profile_Report.html")
    print("✅ Profiling report saved to notebooks/BiztelAI_Chat_Profile_Report.html")


def main():
    df = prepare_data()

    agent_message_stats(df)
    sentiment_analysis(df)
    article_stats(df)
    wordcloud_agent(df, "agent_1")
    wordcloud_agent(df, "agent_2")
    heatmap_turn_ratings(df)

    transcript_summary = summarize_transcripts(df)
    transcript_summary.to_csv("data/chat_transcript_summary.csv", index=False)

    generate_profiling_report(df)

    print("✅ Summary saved to 'data/chat_transcript_summary.csv'")
    print("🔍 Exploratory analysis completed successfully!")

if __name__ == "__main__":
    main()
