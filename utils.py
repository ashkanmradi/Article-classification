import os
import pandas as pd
import matplotlib.pyplot as plt

def Load_Data(data_path):

    TRAIN_DATA_PATH = os.path.join(data_path, 'MINDlarge_train')
    DEVELOPMENT_DATA_PATH = os.path.join(data_path, 'MINDlarge_dev')
    TEST_DATA_PATH = os.path.join(data_path, 'MINDlarge_test')

    train_news_path = os.path.join(TRAIN_DATA_PATH, "news.tsv")
    dev_news_path = os.path.join(DEVELOPMENT_DATA_PATH, "news.tsv")
    test_news_path = os.path.join(TEST_DATA_PATH, "news.tsv")

    train_df = pd.read_table(train_news_path, header=None, names=["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities",])
    test_df = pd.read_table(test_news_path, header=None, names=["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities",])
    

    return train_df, test_df

def Pre_Processing(df_train=None, df_test=None):
    CHOSEN_ARTICLES = ["sports", "finance", "foodanddrink", "health", "travel", "weather", "movies", "music"]

    # remove the rows with Nan
    df_train.dropna(inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    df_test.dropna(inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    print(f'null values in train set after preprocessing:\n{df_train.isnull().sum()}\n')
    print("==============================================\n")

    print(f'null values in test set after preprocessing:\n{df_test.isnull().sum()}\n')
    print("==============================================\n")

    # To avoid scrapping the entire news article with their links, used a combination of the news title and abstract, as the full text to train the model
    df_train["text"] = df_train["title"].astype(str) + df_train["abstract"].astype(str)
    df_train.drop(columns=['title', 'abstract'], inplace=True)

    df_test["text"] = df_test["title"].astype(str) + df_test["abstract"].astype(str)
    df_test.drop(columns=['title', 'abstract'], inplace=True)

    print(f'train shape after dropping nulls: {df_train.shape}\n')
    print(f'test shape after dropping nulls: {df_test.shape}\n')
    print("==============================================\n")

    # There is a category of articles called news! decided to choose 8 categories based on the distribution of category column
    os.makedirs("figs", exist_ok=True)

    ax = df_train.category.value_counts().plot.bar(title="News Categories")
    fig = ax.get_figure()
    fig.savefig("figs/news_categories.png")
    plt.close(fig)  # Close the figure to free memory

    
    df_train = df_train[df_train.category.isin(CHOSEN_ARTICLES)]
    df_test = df_test[df_test.category.isin(CHOSEN_ARTICLES)]

    print(f'train shape after choosing top categories: {df_train.shape}\n')
    print(f'test shape after choosing top categories: {df_test.shape}\n')
    print("==============================================\n")

    return df_train, df_test, CHOSEN_ARTICLES



def display_components(model, word_features, top_display=8):
    # utility for displaying respresentative words per component for topic models
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        top_words_idx = topic.argsort()[::-1][:top_display]
        top_words = [word_features[i] for i in top_words_idx]
        print(" ".join(top_words))