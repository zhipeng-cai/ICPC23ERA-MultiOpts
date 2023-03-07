import os
import time
import pandas as pd
from ranking import bi_gram_cosine_mmr

top_num = 20
input_file = 'predictions/SOTitle/predictions-bs-200.csv'
output_dir = 'predictions/SOTitle/mmr'
output_file = output_dir + '/hypothesis-bs-top20-mmr.csv'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if __name__ == '__main__':
    start_time = time.time()

    predictions = pd.read_csv(input_file)
    predictions.fillna("", inplace=True)
    predictions = predictions.values.tolist()
    deduplicated_predictions = []
    for idx, row in enumerate(predictions):
        candidate_predictions = row[1:]
        candidate_predictions_split = [p.split() for p in candidate_predictions]
        _, ranked_indexs = bi_gram_cosine_mmr(candidate_predictions_split , top_num)
        row_pres = []
        for can_idx in ranked_indexs:
            row_pres.append("[{}]{}".format(can_idx, candidate_predictions[can_idx]))
        deduplicated_predictions.append(row_pres)
        if idx % 100 == 0:
            print('Progress[{}/{}]'.format(idx, len(predictions)), deduplicated_predictions[-1][:5])

    deduplicated_predictions = pd.DataFrame(deduplicated_predictions)
    print(deduplicated_predictions)
    deduplicated_predictions.to_csv(output_file)

    end_time = time.time()
    print("run consume time: ", end_time - start_time)
