from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize

class EvaluationMetrics:

    @staticmethod
    def calculate_bleu(predicted_summary: str, reference_summary: str) -> float:
        """Calculate BLEU score between predicted and reference summary"""
        reference_tokens = [word_tokenize(reference_summary.lower())]
        predicted_tokens = word_tokenize(predicted_summary.lower())
        return sentence_bleu(reference_tokens, predicted_tokens)

    @staticmethod
    def evaluate_sentiment_accuracy(predicted: list, actual: list) -> float:
        """Calculate sentiment classification accuracy"""
        return accuracy_score(actual, predicted)

    @staticmethod
    def batch_bleu(pred_summaries: list, ref_summaries: list) -> float:
        scores = [
            EvaluationMetrics.calculate_bleu(p, r)
            for p, r in zip(pred_summaries, ref_summaries)
        ]
        return sum(scores) / len(scores)
    @staticmethod
    def batch_sentiment_accuracy(pred_sentiments: list, true_sentiments: list) -> float:
        scores = [
            EvaluationMetrics.evaluate_sentiment_accuracy(p, t)
            for p, t in zip(pred_sentiments, true_sentiments)
        ]
        return sum(scores) / len(scores) if scores else 0.0
    @staticmethod
    def batch_accuracy(predictions: list, ground_truth: list) -> float:
        """Calculate overall accuracy for a batch of predictions"""
        return accuracy_score(ground_truth, predictions)
    @staticmethod
    def batch_f1_score(predictions: list, ground_truth: list) -> float:
        """Calculate F1 score for a batch of predictions"""
        from sklearn.metrics import f1_score
        return f1_score(ground_truth, predictions, average='weighted')
    
    @staticmethod
    def batch_precision(predictions: list, ground_truth: list) -> float:
        """Calculate precision for a batch of predictions"""
        from sklearn.metrics import precision_score
        return precision_score(ground_truth, predictions, average='weighted')
    
    @staticmethod
    def batch_recall(predictions: list, ground_truth: list) -> float:
        """Calculate recall for a batch of predictions"""
        from sklearn.metrics import recall_score
        return recall_score(ground_truth, predictions, average='weighted')
    
    @staticmethod
    def batch_confusion_matrix(predictions: list, ground_truth: list) -> dict:
        """Calculate confusion matrix for a batch of predictions"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(ground_truth, predictions)
        return {
            "matrix": cm,
            "labels": list(set(ground_truth + predictions))
        }
    
    @staticmethod
    def batch_classification_report(predictions: list, ground_truth: list) -> str:
        """Generate classification report for a batch of predictions"""
        from sklearn.metrics import classification_report
        return classification_report(ground_truth, predictions, zero_division=0)
    
    @staticmethod
    def batch_roc_auc_score(predictions: list, ground_truth: list) -> float:
        """Calculate ROC AUC score for a batch of predictions"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(ground_truth, predictions, multi_class='ovr', average='weighted')
    @staticmethod
    def batch_pr_auc_score(predictions: list, ground_truth: list) -> float:
        """Calculate Precision-Recall AUC score for a batch of predictions"""
        from sklearn.metrics import average_precision_score
        return average_precision_score(ground_truth, predictions, average='weighted')
    
    @staticmethod
    def batch_jaccard_index(predictions: list, ground_truth: list) -> float:
        """Calculate Jaccard index for a batch of predictions"""
        from sklearn.metrics import jaccard_score
        return jaccard_score(ground_truth, predictions, average='weighted')
    @staticmethod
    def batch_hamming_loss(predictions: list, ground_truth: list) -> float:
        """Calculate Hamming loss for a batch of predictions"""
        from sklearn.metrics import hamming_loss
        return hamming_loss(ground_truth, predictions)
    
    @staticmethod
    def batch_log_loss(predictions: list, ground_truth: list) -> float:
        """Calculate Log Loss for a batch of predictions"""
        from sklearn.metrics import log_loss
        return log_loss(ground_truth, predictions) if len(set(ground_truth)) > 1 else float('inf')
    @staticmethod
    def batch_mean_squared_error(predictions: list, ground_truth: list) -> float:
        """Calculate Mean Squared Error for a batch of predictions"""
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(ground_truth, predictions)
    @staticmethod
    def batch_mean_absolute_error(predictions: list, ground_truth: list) -> float:
        """Calculate Mean Absolute Error for a batch of predictions"""
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(ground_truth, predictions)