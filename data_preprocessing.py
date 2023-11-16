
import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import random
import cv2
import sys
import matplotlib.pyplot as plt
import transformers
import torch

seed = 10
np.random.seed(seed)

# preprocess subject label and data
data_path = '/home/project/data/'
# mapping of label to directory
datasets = {'hmdb51': 'hmdb51', 'kinetics700': 'kinetics700', 'ucf101': 'ucf101'}
img_size = 224

# ------------------------------------

def preprocess_frame(frame):
    # resize image, add additional preprocessing here
    frame = cv2.resize(frame, (img_size, img_size))
    return frame

def extract_k_frames(video_path, num_frames = 5):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # random frame indices
    random_frame_indices = random.sample(range(total_frames), num_frames)

    frames_list = []
    for frame_index in random_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame = preprocess_frame(frame)
            frames_list.append(frame)
    cap.release()

    return np.array(frames_list), random_frame_indices

# ------------------------------------

"""
struct sample_data
key - id of the video

dataset: dataset name
split: train or test
label: label of the action, e.g. kiss, pullup, dance
num_frames: number of frames in the video
frames: numpy array of shape (num_frames, img_size, img_size, 3)
"""

# hmdb51 and ucf101 have common formats
sample_data = {}
for dataset in ['hmdb51', 'ucf101']:
    # first, only get train
    for split in ['train']: #, 'test']:
        classes = [d for d in os.listdir(os.path.join(data_path, datasets[dataset], split))]
        classes = [d for d in classes if os.path.isdir(os.path.join(data_path, datasets[dataset], split, d))]
        
        for cls in classes:
            print(cls)
            # get all videos in the class
            videos = glob.glob(os.path.join(data_path, datasets[dataset], split, cls, '*.avi'))
            print(videos)

            for video_path in videos:
                frames, frame_indices = extract_k_frames(video_path, num_frames = 5)
                sample_id = os.path.basename(video_path)
                # build dict
                if sample_id not in sample_data:
                    sample_data[sample_id] = {
                        'dataset': dataset, 
                        'split': split,
                        'label': cls,
                        'num_frames': frames.shape[0],
                        'frame_indices': frame_indices,
                        'frames': frames,
                        'raw_path': video_path
                    }

# print stats per dataset, split, class, save to csv:
print(f'Stats per dataset, split, class:')
stats_df = pd.DataFrame.from_dict(sample_data, columns=['dataset', 'split', 'label'], orient='index')
stats = stats_df.groupby(['dataset', 'split', 'label']).size().reset_index(name='count')
print(stats)
stats_pivot = stats.pivot_table(index=['dataset', 'split'], columns='label', values='count', fill_value=0)
stats_pivot.to_csv('dataset_stats.csv')

# ------------------------------------
# embed each sample using ViT pretrained on ImageNet21k
# https://huggingface.co/google/vit-base-patch16-224
# we embed each of the k samples randomly extracted from the video and concatenate 
# the embeddings to be used as the final emebedding for the video

from transformers import ViTImageProcessor, ViTForImageClassification

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ckpt = 'google/vit-base-patch16-224'
# load model
img_processor = ViTImageProcessor.from_pretrained(model_ckpt)
model = ViTForImageClassification.from_pretrained(model_ckpt)

# embed each sample
for sample_id in sample_data.keys():
    # get frames
    frames = sample_data[sample_id]['frames']
    # embed each frame
    #frame_embeddings = []
    #for frame in frames:
    #    inputs = img_processor(frame, return_tensors="pt")
    #    outputs = model(**inputs)

    image_batch_transformed = torch.stack(
            [img_processor(frame, return_tensors="pt") for frame in frames]
        )
    new_batch = {"pixel_values": image_batch_transformed.to(device)}
    with torch.no_grad():
        embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        sample_data[sample_id]['embeddings'] = np.concatenate(embeddings.numpy())



# ------------------------------------

# save sample_data to h5
print(f'Saving {len(sample_data)} samples to h5 file...')
h5_path = os.path.join(data_path, 'frames_data.h5')
if not os.path.exists(h5_path):
    f = h5py.File(h5_path, 'a')
    for sample_id in sample_data.keys():
        sample_h5 = f.create_group(sample_id)
        sample_h5.create_dataset('dataset', data=sample_data[sample_id]['dataset'])
        sample_h5.create_dataset('label', data=sample_data[sample_id]['label'])
        sample_h5.create_dataset('split', data=sample_data[sample_id]['split'])
        sample_h5.create_dataset('num_frames', data=sample_data[sample_id]['num_frames'])
        sample_h5.create_dataset('frame_indices', data=sample_data[sample_id]['frame_indices'])
        sample_h5.create_dataset('frames', data=sample_data[sample_id]['frames'])
        sample_h5.create_dataset('raw_path', data=sample_data[sample_id]['raw_path'])
        sample_h5.create_dataset('embeddings', data=sample_data[sample_id]['embeddings'])
    f.close()
else:
    print(f'File {h5_path} already exists. Exiting...')
    sys.exit()
