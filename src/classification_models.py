"""
Module for text classification models using transformers. 
Functions starting with _ are not meant to be called directly but rather are helper functions for use inside the module
"""
from transformers import pipeline


def label_and_score_tweets(tweets, model_id):
    task, model_name = _model_dictionary(model_id)
    pipe = pipeline(task, model=model_name, device="mps")
    results = pipe(tweets, truncation=True)
    labels = []
    scores = []
    for result in results:
        label, score = _translate_label(
            model_id=model_id, label=result["label"], score=result["score"])
        labels.append(label)
        scores.append(score)
    return labels, scores


def _model_dictionary(model_id):
    if model_id == "hate speech 1":
        return "text-classification", "Hate-speech-CNERG/dehatebert-mono-english"
    elif model_id == "hate speech 2":
        return "text-classification", "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    elif model_id == "hate speech 3":
        return "text-classification", "patrickquick/BERTicelli"
    elif model_id == "fake news":
        return "text-classification", "vikram71198/distilroberta-base-finetuned-fake-news-detection"
    elif model_id == "argument detection":
        return "text-classification", "chkla/roberta-argument"
    elif model_id == "sentiment analysis":
        return "text-classification", "finiteautomata/bertweet-base-sentiment-analysis"
    else:
        raise "model id not found found"


def _translate_label(label, score, model_id):
    """ translates the output label of the model into 1 or 0
        and the confidence score into a score between 0 and 1 with closer to 0 values meaning higher confidence in the 0 label
        and closer to 1 values meaning higher confidence in the 1 label.
    """
    if model_id == "hate speech 1":
        label_value = 1 if label == "HATE" else 0
        score = score if label_value == 1 else 1 - score
    elif model_id == "hate speech 2":
        label_value = 0 if label == "normal" else 1
        score = score + 0.5 if label_value == 1 else (score * -1) + 0.5
    elif model_id == "hate speech 3":
        label_value = 1 if label == "OFF" else 0
        score = score if label_value == 1 else 1-score
    elif model_id == "fake news":
        label_value = 1 if label == "LABEL_1" else 0
        score = score if label_value == 1 else 1-score
    elif model_id == "sentiment analysis":
        label_value = label
        score = score
    elif model_id == "argument detection":
        label_value = 1 if label == "ARGUMENT" else 0
        score = score if label_value == 1 else 1 - score
    return label_value, score
