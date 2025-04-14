# Article-classification
Classification of the news articles using classical techniques as well as fine-tuning SOTA models.

## Classical Techniques
Count vectorization, TF-IDF, Non-negative matrix factorization (NMF), latent Dirichlet allocation (LDA), and GloVe were used as the classical approaches to model different articles for classification.
## SOTA Transformers
Here, we used [DistilBert](https://huggingface.co/docs/transformers/en/model_doc/distilbert), [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta), and [XLNet](https://huggingface.co/docs/transformers/en/model_doc/xlnet) as the transformer architectures being fine-tuned. 


## Get Started

### Clone the repository
```
git clone https://github.com/ashkanmradi/Article-classification.git
```

#### Run [main.py](https://github.com/ashkanmradi/Article-classification/blob/main/main.py) and don't forget to modify DATA_PATH according to your own path.

## Notes
- This project was done using [MIND dataset (MIcrosoft News Dataset)](https://msnews.github.io/).
- You can edit the variables at such as NUM_EPOCHS, BATCH_SIZE, MAX_LEN, etc., in the [main.py](https://github.com/ashkanmradi/Article-classification/blob/main/main.py) file. 
- Don't forget to add your own dataset folder path to **DATA_PATH**_variable in [main.py](https://github.com/ashkanmradi/Article-classification/blob/main/main.py).
- Don't forget to edit the first six lines of the Load_Data() function in [utils.py](https://github.com/ashkanmradi/Article-classification/blob/main/utils.py) in order to address your data files.

## The final results on this data are as follow
The transformer architectures were only fine-tuned for five epochs here! Increasing NUM_EPOCHS could lead to a better performance.  
![Confusion Matrix](/figs/Results.png)
