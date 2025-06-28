import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_app.backend_client import get_insights

st.set_page_config(page_title="AI Chat Analyzer", layout="wide")
st.title("🧠 AI Chat Transcript Analyzer (API Powered)")

st.sidebar.header("👥 Paste Chat Transcript")
default_chat = [
    {"agent": "agent_1", "message": "Let’s discuss the article on the new football rule change."},
    {"agent": "agent_2", "message": "Yes, it's causing a lot of debate on ESPN."}
]
chat_input = st.sidebar.text_area("Paste chat JSON here (agent/message format)", value=str(default_chat), height=300)
article_url = st.sidebar.text_input("Optional: Article URL", value="https://www.washingtonpost.com/sports/football-rule-change")

if st.sidebar.button("🔍 Analyze via FastAPI"):
    try:
        chat_list = eval(chat_input) if isinstance(chat_input, str) else chat_input
        payload = {
            "content": chat_list,
            "article_url": article_url
        }

        result = get_insights(payload)

        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            st.success("Analysis received from API! 📊")
            col1, col2 = st.columns(2)
            col1.metric("Agent 1 Messages", result["agent_1_message_count"])
            col2.metric("Agent 2 Messages", result["agent_2_message_count"])

            col1.metric("Agent 1 Sentiment", result["agent_1_sentiment"])
            col2.metric("Agent 2 Sentiment", result["agent_2_sentiment"])

            st.subheader("📝 Summary")
            st.info(result["summary"])

            st.markdown(f"🔗 **Article URL:** [{result['article_url']}]({result['article_url']})")

            st.subheader("☁️ Word Clouds")
            def show_wordcloud(messages, title):
                text = " ".join(messages)
                wc = WordCloud(width=700, height=400, background_color="white").generate(text)
                st.markdown(f"**{title}**")
                st.image(wc.to_array())

            agent1_msgs = [m["message"] for m in chat_list if m["agent"] == "agent_1"]
            agent2_msgs = [m["message"] for m in chat_list if m["agent"] == "agent_2"]
            show_wordcloud(agent1_msgs, "Agent 1")
            show_wordcloud(agent2_msgs, "Agent 2")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
else:
    st.info("Paste a chat transcript in JSON format and click '🔍 Analyze via FastAPI' to get insights.")
    st.warning("Example chat format: [{'agent': 'agent_1', 'message': 'Hello'}, {'agent': 'agent_2', 'message': 'Hi!'}]")

st.sidebar.markdown("""
### About
AI Chat Analyzer is a web application designed to analyze chat transcripts using AI techniques. It provides insights such as message counts, sentiment analysis, and article references to help understand chat dynamics.
### Features
# - **Message Counts**: See how many messages each agent sent.
# - **Sentiment Analysis**: Understand the sentiment of each agent's messages.
# - **Article References**: Automatically infer and display article URLs mentioned in the chat.
# - **Word Clouds**: Visualize the most common words used by each agent.             
This app uses a FastAPI backend to analyze chat transcripts, providing insights like message counts, sentiment analysis, and article references.
### Source Code
[GitHub Repository](https://github.com/yashdew3/Agentic-AI-Chat-Analyzer)
### Contact- Yash Dewangan
For any questions or feedback, please reach out to [Yash Dewangan](https://github.com/yashdew3) on GitHub.
### Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.
### Issues
For any issues or feature requests, please open an issue on the GitHub repository.
### License
This project is licensed under the MIT License.
""")