import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

class ModelHandler:

    def __init__(self, base_dir="models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def download_and_save_flan_t5(self):
        print("🔄 Downloading flan-t5-small...")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model.save_pretrained(os.path.join(self.base_dir, "flan-t5-small"))
        tokenizer.save_pretrained(os.path.join(self.base_dir, "flan-t5-small"))
        print("✅ flan-t5-small saved locally.")

    def download_and_save_cardiffnlp(self):
        print("🔄 Downloading cardiffnlp/twitter-roberta-base-sentiment...")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        model.save_pretrained(os.path.join(self.base_dir, "cardiffnlp-sentiment"))
        tokenizer.save_pretrained(os.path.join(self.base_dir, "cardiffnlp-sentiment"))
        print("✅ Cardiff sentiment model saved locally.")

    def download_all(self):
        self.download_and_save_flan_t5()
        self.download_and_save_cardiffnlp()
