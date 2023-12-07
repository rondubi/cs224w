import h5py
import numpy as np
from typing import List

# we use this weights to capture the similarity.
RELATIONSHIP_WEIGHTS = {
    'is-a': 0.5,
    'similar': 0.25,
    'equal': 1.0
}

# helper function to read h5 file to a dictionary
def read_h5(file_path):
    sample_data = {}
    with h5py.File(file_path, 'r') as f:
        for sample_id in f.keys():
            sample_group = f[sample_id]
            sample_data[sample_id] = {
                'dataset': sample_group['dataset'][()],
                'label': sample_group['label'][()],
                'split': sample_group['split'][()],
                'num_frames': sample_group['num_frames'][()],
                'frame_indices': sample_group['frame_indices'][()],
                'frames': sample_group['frames'][()],
                'raw_path': sample_group['raw_path'][()],
                'embeddings': sample_group['embeddings'][()]
            }
    return sample_data

def related_labels(metavd_df, label : str, dataset : str, relationship : str):
    if relationship not in RELATIONSHIP_WEIGHTS:
        raise ValueError(f'Invalid relationship: {relationship}')
    relations = []
    # get all related labels, is-a, similar, equal
    if relationship == 'is-a':
        # is-a relations are directed, but we treat them as undirected
        relations += metavd_df[(metavd_df['from_action_name'] == label) & (metavd_df['from_dataset'] == dataset) & (metavd_df['relation'] == 'is-a')]['to_action_name'].tolist()
        relations += metavd_df[(metavd_df['to_action_name'] == label) & (metavd_df['to_dataset'] == dataset) & (metavd_df['relation'] == 'is-a')]['from_action_name'].tolist()
    elif relationship == 'similar':
        # similar relations are undirected, so we need to check both directions
        relations += metavd_df[((metavd_df['from_action_name'] == label) & (metavd_df['from_dataset'] == dataset) & (metavd_df['relation'] == 'similar'))]['to_action_name'].tolist()
        relations += metavd_df[((metavd_df['to_action_name'] == label) & (metavd_df['to_dataset'] == dataset) & (metavd_df['relation'] == 'similar'))]['from_action_name'].tolist()
    elif relationship == 'equal':
        # equal relations are undirected, so we need to check both directions
        relations += metavd_df[((metavd_df['from_action_name'] == label) & (metavd_df['from_dataset'] == dataset) & (metavd_df['relation'] == 'equal'))]['to_action_name'].tolist()
        relations += metavd_df[((metavd_df['to_action_name'] == label) & (metavd_df['to_dataset'] == dataset) & (metavd_df['relation'] == 'equal'))]['from_action_name'].tolist()
    else:
        raise ValueError(f'Invalid relationship: {relationship}')
    return relations

def topK_accuracy(metavd_df, dataset : str, true_label : str, pred_labels : List[str], k : int):
    # get all related labels, is-a, similar, equal
    is_a_labels = related_labels(metavd_df, true_label, dataset, 'is-a')
    similar_labels = related_labels(metavd_df, true_label, dataset, 'similar')
    equal_labels = related_labels(metavd_df, true_label, dataset, 'equal')
    if true_label in pred_labels[:k]:
        return 1
    elif len(equal_labels) > 0 and len(set(equal_labels).intersection(set(pred_labels[:k]))) > 0:
        return RELATIONSHIP_WEIGHTS['equal']
    elif len(is_a_labels) > 0 and len(set(is_a_labels).intersection(set(pred_labels[:k]))) > 0:
        return RELATIONSHIP_WEIGHTS['is-a']
    elif len(similar_labels) > 0 and len(set(similar_labels).intersection(set(pred_labels[:k]))) > 0:
        return RELATIONSHIP_WEIGHTS['similar']
    else:
        return 0
    
def topK_accuracy_all(metavd_df, datasets : List[str], true_labels : List[str], pred_labels : List[List[str]], k : int):
    # the number of times where the correct label (or weighted related labels) 
    # is among the top k labels predicted
    correct = 0
    for i in range(len(true_labels)):
        correct += topK_accuracy(metavd_df, datasets[i], true_labels[i], pred_labels[i], k)
    return correct / len(true_labels)

def relation_topK(metavd_df, relation : str, datasets : List[str], true_labels : List[str], pred_labels : List[List[str]], k : int):
    # the average amount of times where a given relation is among the top k labels predicted
    correct = 0
    for i in range(len(true_labels)):
        if relation == 'is-a':
            correct += topK_accuracy(metavd_df, datasets[i], true_labels[i], pred_labels[i], k) == RELATIONSHIP_WEIGHTS['is-a']
        elif relation == 'similar':
            correct += topK_accuracy(metavd_df, datasets[i], true_labels[i], pred_labels[i], k) == RELATIONSHIP_WEIGHTS['similar']
        elif relation == 'equal':
            correct += topK_accuracy(metavd_df, datasets[i], true_labels[i], pred_labels[i], k) == RELATIONSHIP_WEIGHTS['equal']
        else:
            raise ValueError(f'Invalid relationship: {relation}')
    return correct / len(true_labels)

def precision_at_k(metavd_df, dataset : str, true_label : str, pred_labels : List[str], k : int):
    # proportion of relevant items found in top-K recommendations
    count = 0

    # get all related labels, is-a, similar, equal
    is_a_labels = related_labels(metavd_df, true_label, dataset, 'is-a')
    similar_labels = related_labels(metavd_df, true_label, dataset, 'similar')
    equal_labels = related_labels(metavd_df, true_label, dataset, 'equal')

    for i in range(k):
        if pred_labels[i] == true_label:
            count += 1
        elif pred_labels[i] in equal_labels:
            count += RELATIONSHIP_WEIGHTS['equal']
        elif pred_labels[i] in is_a_labels:
            count += RELATIONSHIP_WEIGHTS['is-a']
        elif pred_labels[i] in similar_labels:
            count += RELATIONSHIP_WEIGHTS['similar']

    return count / k

def precision_at_k_all(metavd_df, datasets : List[str], true_labels : List[str], pred_labels : List[List[str]], k : int):
    count = 0
    for i in range(len(true_labels)):
        count += precision_at_k(metavd_df, datasets[i], true_labels[i], pred_labels[i], k)
    return count / len(true_labels)

def NDCG(true_label : str, pred_labels : List[str], k : int):
    if true_label in pred_labels[:k]:
        return 1 / np.log2(pred_labels.index(true_label) + 2)
    else:
        return 0