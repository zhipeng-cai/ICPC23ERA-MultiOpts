import os
import time
import pandas as pd
from rouge import Rouge
from sklearn.cluster import AgglomerativeClustering

beam_num = 200
clusters_nums = [1, 2, 3, 5, 10, 15, 20]
input_file = 'predictions/SOTitle/predictions-bs-200.csv'
output_dir = 'predictions/SOTitle/clustering'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def rouge_vectorizer(texts):
    rouge = Rouge()
    texts_num = len(texts)
    text_vectors = [[0.0] * texts_num for _ in range(texts_num)]

    for i in range(texts_num):
        for j in range(i + 1, texts_num):
            if texts[i] == "" and texts[j] == "":
                similarity = 1.0
            elif texts[i] == "" or texts[j] == "":
                similarity = 0.0
            else:
                try:
                    similarity = rouge.get_scores(texts[i], texts[j])[0]['rouge-l']['f']
                except:
                    similarity = 0.0
            text_vectors[i][j] = 1.0 - similarity
            text_vectors[j][i] = 1.0 - similarity

    return text_vectors


def get_clusters_by_agglomerative(vectors, n_clusters):
    dis_threshold = 0.6
    agglomerative_clf = AgglomerativeClustering(n_clusters=None, distance_threshold=dis_threshold,
                                                affinity='precomputed', linkage='average')
    agglomerative_clf.fit(vectors)
    return_cluster_num = agglomerative_clf.n_clusters_
    if return_cluster_num < n_clusters:
        agglomerative_clf = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        agglomerative_clf.fit(vectors)
    clusters = agglomerative_clf.labels_
    return clusters


def get_diverse_list(candidate_predictions, diverse_limit):
    res = candidate_predictions.groupby(['cluster'])
    res = res['generating_index'].min()
    res = res.sort_values(ascending=True)
    res = res.tolist()
    row_pres = []
    for num in range(len(res)):
        row_pres.append("[{}]{}".format(res[num], candidate_predictions['raw'].iloc[res[num]]))
        if len(row_pres) >= diverse_limit:
            break
    return row_pres


if __name__ == '__main__':
    print("start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    predictions = pd.read_csv(input_file)
    predictions.fillna("", inplace=True)
    predictions = predictions.values.tolist()

    ag_predictions = [[] for _ in range(7)]

    for idx, row in enumerate(predictions):
        rowdata = pd.DataFrame(row[1:])
        rowdata.columns = ['raw']
        rowdata['generating_index'] = range(beam_num)
        X = rouge_vectorizer(rowdata['raw'])
        for clu_idx, clusters_num in enumerate(clusters_nums):
            clusters = get_clusters_by_agglomerative(X, clusters_num)
            rowdata['cluster'] = pd.DataFrame(clusters)
            diverse_list = get_diverse_list(rowdata, clusters_num)
            ag_predictions[clu_idx].append(diverse_list)
        if idx % 100 == 0:
            print('Progress[{}/{}]'.format(idx, len(predictions)), ag_predictions[-1][-1][:5])

    for i in range(len(clusters_nums)):
        deduplicated_predictions = pd.DataFrame(ag_predictions[i])
        print(deduplicated_predictions)
        deduplicated_predictions.to_csv(output_dir + '/hypothesis-bs-top' + str(clusters_nums[i]) + '-clustering.csv')

    print("finish time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
