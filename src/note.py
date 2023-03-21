train_multiknob:
    lib/opt - head
    lib/datasets/dataset/jde - LoadImagesAndLabels, JointDataset_MultiKnob
    lib/datasets/dataset_factory - get_dataset
    lib/trains/train_factory - train_factory
    lib/trains/mot_multiknob - MotTrainer_MultiKnob

track_half_multiknob:
    lib/tracker/multitrack.py - JDETracker - update_hm
    lib/datasets/dataset/jde - LoadImages - set_image_size
