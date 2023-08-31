# Models
Used the hugging face implementation of BERT/DistilBERT and fine tuned them to classify texts.

## bert-base-uncased
From [https://huggingface.co/bert-base-uncased]:
>BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

>Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally masks the future tokens. It allows the model to learn a bidirectional representation of the sentence.
>Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.
>This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences, for instance, you can train a standard classifier using the features produced by the BERT model as inputs.

Datasets used to train bert-base-uncased:
* [https://huggingface.co/datasets/wikipedia] English wikipedia 
* [https://huggingface.co/datasets/bookcorpus] 11,038 unpublished(?) books

Model size: 110M params


## distilbert-base-uncased
From [https://huggingface.co/distilbert-base-uncased]
>DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts using the BERT base model. More precisely, it was pretrained with three objectives:

>Distillation loss: the model was trained to return the same probabilities as the BERT base model.
>Masked language modeling (MLM): this is part of the original training loss of the BERT base model. When taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.
>Cosine embedding loss: the model was also trained to generate hidden states as close as possible as the BERT base model. This way, the model learns the same inner representation of the English language than its teacher model, while being faster for inference or downstream tasks.

Model size: 67M params



# Preprocessing

For review text data, I combined the title and review body for simplicity and sent that through a text_normalization function. (Still tweaking, this is current as of 8/31/23, what else to add?)

Functionally includes:
* Expanding contractions
* Removing punctuation and any formatting characters
* Lowercase

'''

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

'''    

I then one hot encoded the classes and converted that pandas df into a dataset from the datasets library with a train,test, and validation dataset. These are easier to us than pandas with the transformers library for training. 

Next I made a dict of the classes to correspond with an int.

# Tokenizer

For each model I used the corresponding tokenizer from huggingface. I used the same parameters for each.

'''
    tokenizer(text, padding="max_length", truncation=True, max_length=256)
'''

The max_length of 256 and subsequent truncation does cut off some data but my GPU (RTX 3070, 8gb VRam) was not able to handle anything larger with training batch size of 8. The padding is on the right as is recommended.


# Training

'''

    batch_size = 8
    metric_name = "f1"

    args = TrainingArguments(
        f"bert-finetuned-sem_eval-english",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        # push_to_hub=True,
    )

    def multi_label_metrics(
        predictions, labels, threshold=0.5
    ):  # threshold = confidence threshold, important. 0.5 doesnt always work
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # use threshold to turn them into int predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        roc_auc = roc_auc_score(y_true, y_pred, average="micro")
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
        return metrics


    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result


    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )  

    trainer.train()


'''


















