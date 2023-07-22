# Sentiment Analysis of Customer Reviews 🏨
[View the app on HuggingFace](https://huggingface.co/spaces/Uvini/Hotel-Reviews)

An ML model fine-tuned for sentiment analysis of hotel reviews of a selected hotel in Sri Lanka. Moreover, to give added value to the business, a simple app has been designed using [streamlit](https://streamlit.io) to deploy the model. As a test run, the base model before fine-tuning for the custom dataset has been deployed on a space, and you can view it [here](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).

### ▶️ About the base model

[distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

It is a checkpoint of `DistilBERT-base-uncased` model fine-tuned for sentiment analysis on the SST2 dataset which includes movie reviews. Developed by the `Hugging Face community`, the model can be directly used for text classification or further fine-tuned for custom user cases.  

### ▶️ Fine-tuning the model 

Though the initial model is fine-tuned on movie reviews, it is capable of reaching a higher performance on sentiment analysis of most types of reviews, such as hotel reviews, and book reviews.

To make it more flexible for the user scenario, the model is further fine-tuned using customer reviews received by `Heritance Kandalama` a hotel in Sri Lanka. The training process achieved an **accuracy** of **90%**

### ▶️ About training data

The training dataset has **2000 reviews** manually classified as either positive or negative. The dataset is **class balanced**, with each class having 1000 data points.

*Sources: Booking.com, Google Reviews, TripAdvisor, Agoda, Expedia*

### ▶️ Notes

* The Jupiter Notebook `Fine_tuning.ipynb` can be used as the base for fine-tuning a model for similar scenarios on different user cases.
* The `app.py` file contains the source code for the base model deployment and defining a user interface on streamlit.
