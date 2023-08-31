import re
import torch
import numpy as np
import pickle
import pandas as pd
import string
from contractions import contractions_dict

##############################
# Applied to raw text before tokenization
def text_normalization_2(string):

    
    contractions = {key.lower(): value for key, value in contractions_dict.items()}
    for word in string.split():
        if word in contractions:
            string = string.replace(word, contractions[word])

    string = string.lower()
    string = re.sub(r"[^a-zA-Z0-9\s]", " ", string)
    string = re.sub(r"\n", " ", string)
    string = re.sub(r"\s+", " ", string)
    string = string.strip()
    return string



##################################################

def text_normalization_3(string):
    contractions = {key.lower(): value for key, value in contractions_dict.items()}

    fixed_string = string.lower()

    fixed_string = fixed_string.replace(",", " ")

    for word in fixed_string.split():
        if word in contractions:
            fixed_string = fixed_string.replace(word, contractions[word])

    fixed_string = re.sub(r"[^a-zA-Z0-9\s]", " ", fixed_string)

    fixed_string = re.sub(r"\n", " ", fixed_string)
    fixed_string = re.sub(r"\s+", " ", fixed_string)
    fixed_string = fixed_string.lower()
    fixed_string = fixed_string.strip()

    return fixed_string

###############################################3

# Make predictions using a trained model
def predict_class(text, tokenizer, trainer, id2label, CONFIDENCE_THRESHOLD=0.5):
    encoding = tokenizer(text, return_tensors="pt", truncation=True)
    encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}

    outputs = trainer.model(**encoding)

    logits = outputs.logits

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= CONFIDENCE_THRESHOLD)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [
        id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
    ]
    return predicted_labels

##################################################

# Load a pickel file (id2label, label2id)
def load_pickel(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
##################################################    
    
# Penalize multi-category guesses
def multi_cat_guess_penalty(row, muli_cat_guess_penalty=0.85):
    num_guesses = len(row["predicted_class"])

    if num_guesses > 1 and row["correct"] == 1:
        score = (muli_cat_guess_penalty) ** num_guesses

        return score
    else:
        return row["correct"]


##################################################

# read in parquet file and return a dataframe with text normalized

def read_in_and_normalize(file_path):
    df = pd.read_parquet(file_path)
    df = df[["review_title", "review_body", "product_category"]]
    for col in ["review_title", "review_body"]:
        df[col] = df[col].apply(text_normalization_3)

    df["text"] = df["review_title"] + " " + df["review_body"]

    return df

##################################################

# Create a dictionary of class names and probabilities
def n_most_likely_classes(text,tokenizer, trainer, id2label, n=3):
    encoding = tokenizer(text, return_tensors="pt", truncation=True)
    encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}

    outputs = trainer.model(**encoding)

    logits = outputs.logits

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.0)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [
        id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
    ]

    pred_label_probs = list(probs[np.where(probs >= 0.0)].detach().numpy().round(3))
        
    if len(predicted_labels) != len(pred_label_probs):
        raise ValueError(
            "The length of 'classes' and 'probabilities' lists must be the same."
        )

    prob_dict = {class_name: prob for class_name, prob in zip(predicted_labels, pred_label_probs)}
    
    sorted_items = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_n_items = sorted_items[:n] # return this for nestted list
    top_n_dict = dict(top_n_items)

    return top_n_dict

##################################################


# returns 1 if no prediction is made and if a correct prediction is made
def multi_positive_outcome(row):
    if len(row["predicted_class"]) == 0:
        return 1
    elif row["correct"] == 1:
        return 1
    else:
        return 0    


