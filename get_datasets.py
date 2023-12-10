import sys
import fiftyone as fo
import fiftyone.zoo as foz
import traceback
import pandas as pd
import fiftyone.utils.kinetics as kinetics

mnt_path = "/mnt/disks/cs224w-data/data"

dset = sys.argv[1]

"""
for dset in sys.argv[1:]:
    try:
        foz.load_zoo_dataset(dset, split="train", max_samples=10).export(export_dir=f"{mnt_path}/train-{dset}", dataset_type = fo.types.VideoDirectory)
    except:
        traceback.print_exc()
    #try:
    #    foz.load_zoo_dataset(dset, split="validation", max_samples=7000).export(export_dir=f"{mnt_path}/validation-{dset}", dataset_type = fo.types.VideoDirectory)
    #except:
    #    traceback.print_exc() 
    try:
        foz.load_zoo_dataset(dset, split="test", max_samples=10).export(export_dir=f"{mnt_path}/test-{dset}", dataset_type = fo.types.VideoDirectory)
    except:
        traceback.print_exc() 
"""

csv_path = "/mnt/disks/cs224w-data/data/metavd/metavd_v1.csv"
df = pd.read_csv(csv_path)

df = df[(df["from_dataset"] == "kinetics700") | (df["to_dataset"] == "kinetics700")]
df = df[(df["from_dataset"] == "hmdb51") | (df["to_dataset"] == "hmdb51") | (df["to_dataset"] == "ucf101") | (df["from_dataset"] == "ucf101")]

classes = []
classes += df[(df["to_dataset"] == "kinetics700")]["to_action_name"].tolist()
classes += df[(df["from_dataset"] == "kinetics700")]["from_action_name"].tolist()

classes = list(set(classes))
classes = [f"{class_i}".replace("_", " ") for class_i in classes]

kinetics.download_kinetics_split(f"{mnt_path}/{dset}", "train", classes=classes, max_samples=6000, shuffle=True, version='700')
kinetics.download_kinetics_split(f"{mnt_path}/{dset}", "test", classes=classes, max_samples=6000, shuffle=True, version='700')

#kinetics.download_kinetics_split(f"{mnt_path}/{dset}", "train", max_samples=6000, shuffle=True, version='700')
#kinetics.download_kinetics_split(f"{mnt_path}/{dset}", "train", max_samples=6000, shuffle=True, version='700')

