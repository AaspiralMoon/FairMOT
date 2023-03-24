train_multiknob:
    lib/opt - head
    lib/datasets/dataset/jde - LoadImagesAndLabels, JointDataset_MultiKnob
    lib/datasets/dataset_factory - get_dataset
    lib/trains/train_factory - train_factory
    lib/trains/mot_multiknob - MotTrainer_MultiKnob

track_half_multiknob:
    lib/tracker/multitrack.py - JDETracker - update_hm
    lib/datasets/dataset/jde - LoadImages - set_image_size


Plan:
1. Finish multi-res training of half model
2. Re-train multiknob model
3. Finish multires training of full and quarter model
4. Re-generate train and result folder
5. Re-train multiknob model
6. Add QP
7  ...