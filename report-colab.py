import statistics

filenames = [
    'user-ct-test-collection-01.txt',
    'user-ct-test-collection-02.txt',
    'user-ct-test-collection-03.txt',
    'user-ct-test-collection-04.txt',
    'user-ct-test-collection-05.txt',
    'user-ct-test-collection-06.txt',
    'user-ct-test-collection-07.txt',
    'user-ct-test-collection-08.txt',
    'user-ct-test-collection-09.txt',
    'user-ct-test-collection-10.txt' 
]


# すべてのデータのクエリを取得
all_query = []
# ユーザのリスト
user_list  =[]
# クエリの度数分布用辞書
query_dic = {}
# ユーザのクリック数用辞書
user_dic = {}
all_query_count = 0
unique_query_count = 0
unique_user_count = 0

for filename in filenames:
    with open('/content/drive/MyDrive/data/practice/' + filename, encoding="utf-8") as f:
        for line in f:
            line = line.split('\t')
            # 改行コードや空白を取り除く
            line = [item for item in line if item != '']
            line = [item for item in line if item != '\n']

            query= line[1]
            user = line[0]
            all_query.append(query)
            user_list.append(user)
            # count query frequency
            if query in query_dic.keys():
                query_dic[query] += 1
            else:
                query_dic[query] = 1
            # count click s of each user
            if len(line) == 5:
                if user in user_dic.keys():
                    user_dic[user] += 1
                else:
                    user_dic[user] = 1

    all_query_count += len(all_query)
    unique_query = set(all_query)

    unique_query_count += len(unique_query)
    unique_user_count += len(set(user_list))

print(all_query_count)
# ユニークなクエリの数
print(unique_query_count)
# ユニークなユーザの数
print(unique_user_count)
# query frequencyの中央値と平均値
print(statistics.mean(query_dic.values()))
print(statistics.median(query_dic.values()))
# ユーザごとのクリック数平均
print(statistics.mean(user_dic.values()))