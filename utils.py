import h5py
import numpy as np
from typing import List

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

def topK_accuracy(true_label : str, pred_labels : List[str], k : int):
    # pred_labels must be sorted in descending order of confidence!
    if true_label in pred_labels[:k]:
        return 1
    else:
        return 0
    
def topK_accuracy_all(true_labels : List[str], pred_labels : List[str], k : int):
    correct = 0
    for i in range(len(true_labels)):
        correct += topK_accuracy(true_labels[i], pred_labels[i], k)
    return correct / len(true_labels)

def precision_at_k(true_label : str, pred_labels : List[str], k : int):
    # pred_labels must be sorted in descending order of confidence!
    if true_label in pred_labels[:k]:
        return 1
    else:
        return 0

def NDCG(true_label : str, pred_labels : List[str], k : int):
    if true_label in pred_labels[:k]:
        return 1 / np.log2(pred_labels.index(true_label) + 2)
    else:
        return 0