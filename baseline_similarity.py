from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import tqdm

LOCAL = False
top_K = 10
path =  '/mnt/disks/cs224w-data/data/frames_data.h5' if not LOCAL else 'frames_data_small.h5'
metavd_path = '/mnt/disks/cs224w-data/data/metavd/metavd_v1.csv' if not LOCAL else 'metavd/metavd_v1.csv'

def main():
    # load data
    data_dict = read_h5(path)
    sample_ids = list(data_dict.keys())
    metavd_df = pd.read_csv(metavd_path)

    # randomly select one frame to visually inspect as sanity check
    view_sample_id = np.random.choice(list(data_dict.keys()))
    print(f'Inspecting sample {view_sample_id}...')
    print(f'Label: {data_dict[view_sample_id]["label"]}')
    print(f'Dataset: {data_dict[view_sample_id]["dataset"]}')
    frame = data_dict[view_sample_id]['frames'][0]
    plt.imshow(frame)
    plt.show()

    # only keep relevant data for memory efficiency
    for sample_id in sample_ids:
        if data_dict[sample_id]['num_frames'] == 5:
            data_dict[sample_id] = {
                'label': data_dict[sample_id]['label'].decode('ASCII'),
                'dataset': data_dict[sample_id]['dataset'].decode('ASCII'),
                'split': data_dict[sample_id]['split'].decode('ASCII'),
                'embeddings': data_dict[sample_id]['embeddings']
            }
        else:
            print(f'Warning: {sample_id} has {data_dict[sample_id]["num_frames"]} frames, skipping...')
            del data_dict[sample_id]

    print(f'Loaded {len(data_dict)} samples.')
    
    sample_ids = list(data_dict.keys())
    embeddings = [data_dict[sample_id]['embeddings'] for sample_id in sample_ids]
    embeddings = np.array(embeddings)
    labels = np.array([data_dict[sample_id]['label'] for sample_id in sample_ids])
    datasets = [data_dict[sample_id]['dataset'] for sample_id in sample_ids]

    # select 20% of data for testing
    #test_data_dict = np.random.choice(list(data_dict.keys()), int(len(data_dict) * 0.2), replace = False)
    # however don't exclude the test data from possible match candidates (similar to GNN edge prediction on inference, nodes aren't removed)

    # --------------------------------

    # compute similarity predictions
    print('Computing similarity predictions...')
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    similarity_results = defaultdict(dict)
    for i, sample_id in enumerate(tqdm.tqdm(sample_ids)):
        similar_indices = np.argsort(similarity_matrix[i])[::-1][1:top_K + 1]
        similarity_results[sample_id]['true_label'] = labels[i]
        similarity_results[sample_id]['pred_labels'] = [label for label in labels[similar_indices]]
        similarity_results[sample_id]['pred_scores'] = similarity_matrix[i][similar_indices]
        similarity_results[sample_id]['pred_ids'] = [sample_ids[j] for j in similar_indices]

    # --------------------------------

    print('Computing metrics...')
    true_labels = [similarity_results[sample_id]['true_label'] for sample_id in similarity_results]
    pred_labels = [similarity_results[sample_id]['pred_labels'] for sample_id in similarity_results]

    print('---- Accuracy Metrics ----')
    topK_acc = topK_accuracy_all(metavd_df, datasets, true_labels, pred_labels, top_K)
    print(f'Top-{top_K} accuracy: {topK_acc}')

    top_5_acc = topK_accuracy_all(metavd_df, datasets, true_labels, pred_labels, 5)
    print(f'Top-{5} accuracy: {top_5_acc}')
    
    top_1_acc = topK_accuracy_all(metavd_df, datasets, true_labels, pred_labels, 1)
    print(f'Top-1 accuracy: {top_1_acc}')

    topK_is_a = relation_topK(metavd_df, 'is-a', datasets, true_labels, pred_labels, top_K)
    print(f'Top-{top_K} is-a: {topK_is_a}')
    topK_similar = relation_topK(metavd_df, 'similar', datasets, true_labels, pred_labels, top_K)
    print(f'Top-{top_K} similar: {topK_similar}')
    topK_equal = relation_topK(metavd_df, 'equal', datasets, true_labels, pred_labels, top_K)
    print(f'Top-{top_K} equal: {topK_equal}')

    print('---- Ranking Precision Metrics ----')
    topK_prec = precision_at_k_all(metavd_df, datasets, true_labels, pred_labels, top_K)
    print(f'Top-{top_K} precision: {topK_prec}')

if __name__ == '__main__':
    main()