import pandas as pd
from rouge import Rouge
# from myrouge.rouge import Rouge
# refer to the pr description replication package

# print the detailed rouge metric
def print_metric(topk_metrics):
    def r(x):
        return round(x * 100, 4)

    print('recall-----')
    # print recall
    for avg_score in topk_metrics:
        print('rouge-1: {:.4}\t\trouge-2: {:.4}\t\trouge-l: {:.4}'.format(r(avg_score['rouge-1']['r']),
                                                                          r(avg_score['rouge-2']['r']),
                                                                          r(avg_score['rouge-l']['r'])))
    print('precision-----')
    # print precision
    for avg_score in topk_metrics:
        print('rouge-1: {:.4}\t\trouge-2: {:.4}\t\trouge-l: {:.4}'.format(r(avg_score['rouge-1']['p']),
                                                                          r(avg_score['rouge-2']['p']),
                                                                          r(avg_score['rouge-l']['p'])))
    print('f1-----')
    # print f1
    for avg_score in topk_metrics:
        print('rouge-1: {:.4}\t\trouge-2: {:.4}\t\trouge-l: {:.4}'.format(r(avg_score['rouge-1']['f']),
                                                                          r(avg_score['rouge-2']['f']),
                                                                          r(avg_score['rouge-l']['f'])))


# find the max rouge hypotheses and calculate the rouge metric
def max_rouge(list_of_hypotheses, list_of_references, whether_drop_index_number):
    rouge = Rouge()
    max_hypotheses = []

    for multi_hypotheses, ref in zip(list_of_hypotheses, list_of_references):
        max_score = 0.0
        max_sentence = None
        for hyp in multi_hypotheses:
            if hyp == "":
                continue
            # drop the index number
            if whether_drop_index_number:
                hyp = hyp[hyp.index(']') + 1:]
            try:
                score = rouge.get_scores(hyp, ref)[0]['rouge-l']['f']
            except:
                score = 0.0
                hyp = ""
            if score >= max_score:
                max_score = score
                max_sentence = hyp

        max_hypotheses.append(max_sentence)

    avg_score = rouge.get_scores(max_hypotheses, list_of_references, avg=True, ignore_empty=True)

    return avg_score


# pass the filename and then return the rouge metric
def get_rouge_by_file(prediction_file, label_file, drop_index_number):
    evaluation_labels = pd.read_csv(label_file)
    evaluation_labels = evaluation_labels['target'].tolist()
    evaluation_predictions = pd.read_csv(prediction_file)
    evaluation_predictions.fillna("", inplace=True)
    evaluation_predictions1 = evaluation_predictions.iloc[:, 1:2].values.tolist()
    evaluation_predictions2 = evaluation_predictions.iloc[:, 1:3].values.tolist()
    evaluation_predictions3 = evaluation_predictions.iloc[:, 1:4].values.tolist()
    evaluation_predictions5 = evaluation_predictions.iloc[:, 1:6].values.tolist()
    evaluation_predictions10 = evaluation_predictions.iloc[:, 1:11].values.tolist()
    evaluation_predictions15 = evaluation_predictions.iloc[:, 1:16].values.tolist()
    evaluation_predictions20 = evaluation_predictions.iloc[:, 1:21].values.tolist()

    avg_score1 = max_rouge(evaluation_predictions1, evaluation_labels, drop_index_number)
    avg_score2 = max_rouge(evaluation_predictions2, evaluation_labels, drop_index_number)
    avg_score3 = max_rouge(evaluation_predictions3, evaluation_labels, drop_index_number)
    avg_score5 = max_rouge(evaluation_predictions5, evaluation_labels, drop_index_number)
    avg_score10 = max_rouge(evaluation_predictions10, evaluation_labels, drop_index_number)
    avg_score15 = max_rouge(evaluation_predictions15, evaluation_labels, drop_index_number)
    avg_score20 = max_rouge(evaluation_predictions20, evaluation_labels, drop_index_number)

    print_metric([avg_score1, avg_score2, avg_score3, avg_score5, avg_score10, avg_score15, avg_score20])


def cal_rouge_for_topk(prediction_dir, label_file, drop_index_number):
    evaluation_labels = pd.read_csv(label_file)
    evaluation_labels = evaluation_labels['target'].tolist()
    evaluation_predictions1 = pd.read_csv(prediction_dir + '/hypothesis-bs-top1-agglomerative.csv')
    evaluation_predictions1.fillna("", inplace=True)
    evaluation_predictions1 = evaluation_predictions1.iloc[:, 1:2].values.tolist()
    evaluation_predictions2 = pd.read_csv(prediction_dir + '/hypothesis-bs-top2-agglomerative.csv')
    evaluation_predictions2.fillna("", inplace=True)
    evaluation_predictions2 = evaluation_predictions2.iloc[:, 1:3].values.tolist()
    evaluation_predictions3 = pd.read_csv(prediction_dir + '/hypothesis-bs-top3-agglomerative.csv')
    evaluation_predictions3.fillna("", inplace=True)
    evaluation_predictions3 = evaluation_predictions3.iloc[:, 1:4].values.tolist()
    evaluation_predictions5 = pd.read_csv(prediction_dir + '/hypothesis-bs-top5-agglomerative.csv')
    evaluation_predictions5.fillna("", inplace=True)
    evaluation_predictions5 = evaluation_predictions5.iloc[:, 1:6].values.tolist()
    evaluation_predictions10 = pd.read_csv(prediction_dir + '/hypothesis-bs-top10-agglomerative.csv')
    evaluation_predictions10.fillna("", inplace=True)
    evaluation_predictions10 = evaluation_predictions10.iloc[:, 1:11].values.tolist()
    evaluation_predictions15 = pd.read_csv(prediction_dir + '/hypothesis-bs-top15-agglomerative.csv')
    evaluation_predictions15.fillna("", inplace=True)
    evaluation_predictions15 = evaluation_predictions15.iloc[:, 1:16].values.tolist()
    evaluation_predictions20 = pd.read_csv(prediction_dir + '/hypothesis-bs-top20-agglomerative.csv')
    evaluation_predictions20.fillna("", inplace=True)
    evaluation_predictions20 = evaluation_predictions20.iloc[:, 1:21].values.tolist()
    avg_score1 = max_rouge(evaluation_predictions1, evaluation_labels, drop_index_number)
    avg_score2 = max_rouge(evaluation_predictions2, evaluation_labels, drop_index_number)
    avg_score3 = max_rouge(evaluation_predictions3, evaluation_labels, drop_index_number)
    avg_score5 = max_rouge(evaluation_predictions5, evaluation_labels, drop_index_number)
    avg_score10 = max_rouge(evaluation_predictions10, evaluation_labels, drop_index_number)
    avg_score15 = max_rouge(evaluation_predictions15, evaluation_labels, drop_index_number)
    avg_score20 = max_rouge(evaluation_predictions20, evaluation_labels, drop_index_number)
    print_metric([avg_score1, avg_score2, avg_score3, avg_score5, avg_score10, avg_score15, avg_score20])


if __name__ == '__main__':
    prediction_file = 'predictions/SOTitle/predictions-bs-200.csv'
    label_file = 'datasets/SOTitle/test.csv'
    print(prediction_file)
    get_rouge_by_file(prediction_file, label_file, False)
    # cal_rouge_for_topk(prediction_file, label_file, True)
