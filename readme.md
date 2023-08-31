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

I think this is a fairly standard implementation of a transformers library training loop. Based on [https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb]



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

    trainer.evaluate()


'''

# Evaluation


## Amazon reviews classified to product category
Model: bert-base-uncased

Training size: 100k

Training time: 2 hrs

Number of classes: 30

Results on unseen test data:

results_df and image of graph


Model: bert-base-uncased

Training size: 10k

Training time: 20 mins

Number of classes: 30

Results on unseen data:

##############


## Movie plots classified to genre
Model: distilbert-base-uncased

Training size: ###########

Training time: 12 mins

Number of classes:  100

Results on unseen test data:

#########


## News headlines classified to news category

Model: bert-base-uncased

Training size: #########

Training time: 24 mins

Number of classes:  41

Results on unseen test data:

#########




'+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|   threshold |   percent_correct |   percent_correct_discount |   percent_correct_non_preds |   percent_non_preds |   percent_wrong_preds |\n+=============+===================+============================+=============================+=====================+=======================+\n|        0.05 |             76.08 |                      59.33 |                       76.08 |                0    |                 23.92 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.1  |             66.2  |                      58.68 |                       66.42 |                0.22 |                 33.58 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.15 |             60.78 |                      56.37 |                       64.12 |                3.34 |                 35.88 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.2  |             57.62 |                      54.76 |                       64.82 |                7.2  |                 35.18 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.25 |             54.5  |                      52.67 |                       66.14 |               11.64 |                 33.86 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.3  |             52    |                      50.88 |                       68.12 |               16.12 |                 31.88 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.35 |             49.96 |                      49.32 |                       70.48 |               20.52 |                 29.52 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.4  |             47.66 |                      47.38 |                       72.94 |               25.28 |                 27.06 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.45 |             46.02 |                      45.94 |                       75.3  |               29.28 |                 24.7  |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.5  |             44.58 |                      44.55 |                       78.1  |               33.52 |                 21.9  |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.55 |             42.9  |                      42.89 |                       80.68 |               37.78 |                 19.32 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.6  |             41    |                      41    |                       83.36 |               42.36 |                 16.64 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.65 |             38.44 |                      38.44 |                       85.74 |               47.3  |                 14.26 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.7  |             35.84 |                      35.84 |                       88.04 |               52.2  |                 11.96 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.75 |             33.66 |                      33.66 |                       90.34 |               56.68 |                  9.66 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.8  |             30.62 |                      30.62 |                       92.46 |               61.84 |                  7.54 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.85 |             25.5  |                      25.5  |                       95.22 |               69.72 |                  4.78 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.9  |             16.7  |                      16.7  |                       97.94 |               81.24 |                  2.06 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+\n|        0.95 |              1.78 |                       1.78 |                       99.84 |               98.06 |                  0.16 |\n+-------------+-------------------+----------------------------+-----------------------------+---------------------+-----------------------+'









