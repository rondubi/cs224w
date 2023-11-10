import sys
import fiftyone as fo
import fiftyone.zoo as foz

for dset in sys.argv[1:]:
    try:
        foz.load_zoo_dataset(dset, split="train").export(export_dir=f"train-{dset}", dataset_type = fo.types.VideoDirectory)
    except:
        pass
    try:
        foz.load_zoo_dataset(dset, split="validation").export(export_dir=f"validation-{dset}", dataset_type = fo.types.VideoDirectory)
    except:
        pass
    try:
        foz.load_zoo_dataset(dset, split="test").export(export_dir=f"test-{dset}", dataset_type = fo.types.VideoDirectory)
    except:
        pass

