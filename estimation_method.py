
# NaiveBayes
# 入力: 問題テキスト(df)と正解難易度ラベル
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
label_df = pd.read_csv("label_df.csv", sep = ",")
input_df = pd.read_csv("input.csv", sep = ",")

batch_df = pd.read_csv("batch_100.csv", sep = ",")
#label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
label_df = label_df.set_index('id')

def naivebayes(label_df, train_task, test_task):
    '''
    X_train, X_test, y_train, y_test = train_test_split(
    label_df['title'], 
    label_df['true_label'],
    test_size = 0.4,
    random_state = 1
    )
    '''

    X_train = label_df['title'][train_task]
    X_test = label_df['title'][test_task]
    y_train = label_df['difficulty'][train_task]
    y_test = label_df['difficulty'][test_task]


    # test_list: テストデータのIDリスト
    test_list = list(X_test.index)
    # test_list: 訓練データのIDリスト
    train_list = list(X_train.index)

    count_vector = CountVectorizer(stop_words = 'english')
    training_data = count_vector.fit_transform(X_train)
    testing_data = count_vector.transform(X_test)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)
    predictions = naive_bayes.predict(testing_data)
    '''
    予測完了
    '''
    # テストデータ X_testに対する分類結果
    prob_list = naive_bayes.predict_proba(testing_data)

    # test_list: テストデータのIDリスト
    test_list = list(X_test.index)
    # test_list: 訓練データのIDリスト
    train_list = list(X_train.index)
    difficulty_dic = {}

    for i in range(len(test_list)):
        task = test_list[i]
        difficulty_label = np.argmax(prob_list[i])
        difficulty_dic[task] = difficulty_label

    return difficulty_dic


# print("Training dataset: ", X_train.shape[0])
# print("Test dataset: ", X_test.shape[0])
