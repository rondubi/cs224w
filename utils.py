import h5py
import numpy as np

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