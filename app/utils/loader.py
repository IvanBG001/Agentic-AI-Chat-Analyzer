import json
import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        with open(self.filepath, 'r') as file:
            data = json.load(file)

        records = []
        for conv_id, conversation in data.items():
            for message in conversation["content"]:
                records.append({
                    "conversation_id": conv_id,
                    "article_url": conversation.get("article_url"),
                    "config": conversation.get("config"),
                    "agent": message.get("agent"),
                    "message": message.get("message"),
                    "sentiment": message.get("sentiment"),
                    "knowledge_source": message.get("knowledge_source"),
                    "turn_rating": message.get("turn_rating"),
                    "agent_rating": conversation.get("conversation_rating", {}).get(message.get("agent"))
                })
        df = pd.DataFrame(records)
        return df
