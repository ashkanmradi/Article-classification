from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from tqdm import tqdm
import spacy
from common.article_classification_dataset import ArticleClassificationDataProcessor, ArticleClassificationDataSet
from common.article_classification_model import ArticleClassifier
from common.plots import pca_plots
import utils  



CACHE_DIR = TemporaryDirectory().name
NUM_EPOCHS = 1
BATCH_SIZE = 32
NUM_GPUS = torch.cuda.device_count()
print("==============================================")
print(f'Available GPUs: {NUM_GPUS}')
print("==============================================")
MAX_LEN = 100
MODEL_NAMES = ["distilbert-base-uncased", "roberta-base", "xlnet-base-cased"]
MODEL_RESULTS = dict()
LABEL_COL = "category"
TEXT_COL = "text"


DATA_PATH = '/home/ashkan2/scratch/ashkan2/Data/NLP_dataset'
df_train, df_test = utils.Load_Data(DATA_PATH)

print(f'train shape: {df_train.shape}')
print(f'test shape: {df_test.shape}')
print("==============================================")

df_train = df_train[["title", "abstract", "category"]]
df_test = df_test[["title", "abstract", "category"]]

print(f'null values in train set:\n{df_train.isnull().sum()}')
print("==============================================")
print(f'null values in test set:\n{df_test.isnull().sum()}')
print("==============================================")

# Data preprocessing
df_train, df_test, chosen_articles = utils.Pre_Processing(df_train=df_train, df_test=df_test)

print(f'number of news in each train category:\n{df_train[LABEL_COL].value_counts()}')
print(f'number of news in each test category:\n{df_test[LABEL_COL].value_counts()}')

# NLP preprocessing on train df
nlp = spacy.load("en_core_web_md")

def Tokens_Processing(doc, model=nlp):
  return [x.lemma_.lower() for x in nlp(doc) if (x.is_alpha)&
          (not x.like_url)&(not x.is_punct)&(not x.is_stop)]

news_data = [Tokens_Processing(str(d)) for d in df_train['text']]

cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False,min_df=2) # tokenizer=lambda doc: doc: This tells the vectorizer not to do any tokenization.
tfidf = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False,min_df=2) # min_df=2: ignore words that appear in fewer than 2 documents. reduce noise and dimensionality.

cv_vecs = cv.fit_transform(news_data).toarray()
tfidf_vecs = tfidf.fit_transform(news_data).toarray()

cv_vocab = cv.vocabulary_
tfidf_vocanb = tfidf.vocabulary_
vocab = cv_vocab 
print("==============================================")
print(f"vocab size - cv: {len(cv_vocab)} ------ vocab size - tfidf: {len(tfidf_vocanb)}")

print(f'cv shape: {cv_vecs.shape}')
print(f'tfidf shape: {tfidf_vecs.shape}')
print("==============================================")


# Top 'n' words for each news category articles (Counts and TF-IDF)
top_words = 10

categories_rows_indx = []
df_train.reset_index(drop=True, inplace=True)  # ensures clean 0-based indexing

for category in chosen_articles:
    categories_rows_indx.append((category, df_train[df_train['category'] == category].index.to_list()))


for vectorizer, vecs  in [(cv, cv_vecs), (tfidf, tfidf_vecs)]:
    print(f"Top {top_words} words using {vectorizer.__class__.__name__}")
    for category, indx in categories_rows_indx:    
        # sum counts
        s_sum = vecs[indx].sum(axis=0)
        # sort arguments
        s_sorted = np.argsort(s_sum)
        print(f"Category: {category}:")
        print([vectorizer.get_feature_names_out()[x] for x in s_sorted[-top_words:]])
        print("=========================================================================")
    print("\n")


# Analyzing with Topic Modeling
n_components=len(chosen_articles)
nmf = NMF(n_components=n_components)
lda = LatentDirichletAllocation(n_components=n_components)

# tfidf for nmf
nmf_vecs = nmf.fit_transform(tfidf_vecs)
nmf_words = nmf.components_.T

# count for lda
lda_vecs = lda.fit_transform(cv_vecs)
lda_words = lda.components_.T

# Topic model performance
print('NMF Reconstruction err:', nmf.reconstruction_err_)
print('LDA ELBO:', lda.bound_)
print("==============================================")

print('Displaying top 10 words in each topic using NMF(tfidf): ')
utils.display_components(nmf, tfidf.get_feature_names_out())
print("==============================================")

print('Displaying top 10 words in each topic using LDA(CountVectorizer): ')
utils.display_components(lda, cv.get_feature_names_out())
print("==============================================")



# get vectors for vocab
glove_vecs = np.zeros(shape=(len(vocab), 300))
for k, v in vocab.items():
    glove_vecs[v] = nlp(k).vector


vec_list = [
            ("CV", cv_vecs),
            ("TFIDF", tfidf_vecs),
            ("NMF", nmf_words),
            ("LDA", lda_words),
            ("Glove", glove_vecs)
           ]


for name, vecs in vec_list:
    if name in ['NMF', 'LDA', 'Glove']:
        continue
    pca_plots(df_train, vecs, name, chosen_articles)


# Fitting SVM models for TFIDF, CV, NMF, LDA and Glove vectors
X_train = df_train['text']
y_train = df_train['category']
X_test = df_test['text']
y_test = df_test['category']


n_components=len(chosen_articles)
svc = LinearSVC()
tfidf = TfidfVectorizer(tokenizer=Tokens_Processing, min_df=2)
cv = CountVectorizer(tokenizer=Tokens_Processing, min_df=2)
nmf = NMF(n_components=n_components)
lda = LatentDirichletAllocation(n_components=n_components)

# TFIDF featureset
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

# CV featureset
cv_train = cv.fit_transform(X_train)
cv_test = cv.transform(X_test)

# NMF featureset
nmf_train = nmf.fit_transform(tfidf_train)
nmf_test = nmf.transform(tfidf_test)

# Glove featureset
glove_train = np.concatenate([nlp(doc).vector.reshape(1, -1) for doc in X_train])
glove_test = np.concatenate([nlp(doc).vector.reshape(1, -1) for doc in X_test])


# fit/predict on full dataset
svc = LinearSVC(max_iter=10000)
for pair in [(tfidf_train, tfidf_test, "tfidf"), (cv_train, cv_test, "cv"), (nmf_train, nmf_test, "nmf"), (glove_train, glove_test, "glove")]:
    svc.fit(pair[0], y_train)
    preds_test = svc.predict(pair[1])
    preds_train = svc.predict(pair[0])
    acc_train = accuracy_score(y_train, preds_train)
    acc_test = accuracy_score(y_test, preds_test)
    

    class_report = classification_report(y_test, preds_test, output_dict=True)

    MODEL_RESULTS[pair[2]] = {
        "Test Set Accuracy": round(acc_test * 100, 2),
        "f1-score(macro avg)": round(class_report["macro avg"]["f1-score"] * 100, 2),
        "precision": round(class_report["weighted avg"]["precision"] * 100, 2),
        "recall": round(class_report["weighted avg"]["recall"] * 100, 2),
        "f1-score(weighted avg)": round(class_report["weighted avg"]["f1-score"] * 100, 2),
    }
    
    print(f"{pair[2]} Train acc : ", round(acc_train * 100, 2))
    print(f"{pair[2]} Test acc : ", round(acc_test * 100, 2))
    print("==============================================")



pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.max_rows', None) # Show all rows (optional)
pd.set_option('display.max_colwidth', None) # Prevent truncation of column content
pd.set_option('display.width', None) # Avoid width-based line wrapping

print(pd.DataFrame(MODEL_RESULTS).transpose())
print("================================================================")

# Evaluation of Transformer Based Models
label_encoder = LabelEncoder()
df_train[LABEL_COL] = label_encoder.fit_transform(df_train[LABEL_COL])
df_test[LABEL_COL] = label_encoder.transform(df_test[LABEL_COL])


print("Below models will be fine-tuned ... ")
print(MODEL_NAMES)
print("==============================================")


for name in tqdm(MODEL_NAMES, disable=True):
    # Steps
    # 1. Dataframe --> PyTorch Dataset
    # 2. Dataset --> PyTorch Dataloader
    processor = ArticleClassificationDataProcessor(
        model_name=str(name),
        to_lower=name.endswith("uncased"),
        batch_size=BATCH_SIZE, 
        num_gpus=NUM_GPUS,
        cache_dir=CACHE_DIR
    )
    
    train_dataset = processor.create_dataset_from_dataframe(df_train, TEXT_COL, LABEL_COL, max_len=MAX_LEN)
    train_dataloader = processor.create_dataloader_from_dataset(train_dataset, shuffle=True)
    
    test_dataset = processor.create_dataset_from_dataframe(df_test, TEXT_COL, LABEL_COL, max_len=MAX_LEN)
    test_dataloader = processor.create_dataloader_from_dataset(test_dataset, shuffle=False)


    # fine-tune the classifier using the article dataloader
    num_labels = len(np.unique(df_train[LABEL_COL]))
    classifier = ArticleClassifier(model_name=name, num_labels=num_labels, cache_dir=CACHE_DIR)
    classifier.fit(train_dataloader, num_epochs=NUM_EPOCHS, num_gpus=NUM_GPUS, verbose=False)

    # predict on the test set
    preds = classifier.predict(test_dataloader, num_gpus=NUM_GPUS, verbose=True)

    # evaluate the model accuracy 
    accuracy = accuracy_score(df_test[LABEL_COL], preds)
    class_report = classification_report(df_test[LABEL_COL], preds, target_names=label_encoder.classes_, output_dict=True)

    # save results
    MODEL_RESULTS[name] = {
        "Test Set Accuracy": round(accuracy * 100, 2),
        "f1-score(macro avg)": round(class_report["macro avg"]["f1-score"] * 100, 2),
        "precision": round(class_report["weighted avg"]["precision"] * 100, 2),
        "recall": round(class_report["weighted avg"]["recall"] * 100, 2),
        "f1-score(weighted avg)": round(class_report["weighted avg"]["f1-score"] * 100, 2),
    }


print("================================================================")
print(pd.DataFrame(MODEL_RESULTS).transpose())
print("================================================================")