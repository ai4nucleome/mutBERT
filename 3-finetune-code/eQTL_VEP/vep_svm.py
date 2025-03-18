import random
import time
from os import path as osp

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

DIST_TO_TSS = [[0, 30_000], [30_000, 100_000], [100_000, np.infty]]
USE_TISSUE = [True]  # used as another for loop for fitting SVM, whether to use tissue embed or not
Cs = [1, 5, 10]  # for loop in fitting SVM, inverse of L2 penalty (sklearn hyperparam)
PATH_TO_OUTPUTS = "output"

def dataset_nan_filter(data: dict, data_key: str) -> dict:
    """Filter any items that have NaN in embedding within TSS bucket"""
    mask_out = torch.logical_or(
        torch.any(data[data_key].isnan(), dim=1),
        torch.any(data[f"{data_key}"].isnan(), dim=1)
    )
    
    new_data = dict()
    for data_key in data.keys():
        new_data[data_key] = data[data_key][~mask_out]

    return new_data

def dataset_tss_filter(data: dict, min_distance: int, max_distance: int) -> dict:
    """Filter the data to items that fall within TSS bucket"""
    distance_mask = ((data["distance_to_nearest_tss"] >= min_distance) 
                     & (data["distance_to_nearest_tss"] <= max_distance))
    new_data = dict()
    for data_key in data.keys():
        new_data[data_key] = data[data_key][distance_mask]

    return new_data


model_dict = {
    "mutbert": dict(
        embed_path="mutbert_seqlen=2k",
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        key="concat_avg_ws",
    ),
    "dnabert2": dict(
        embed_path="dnabert2_seqlen=2k",
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        key="concat_avg_ws",
    ),
    "NT_50_multi": dict(
        embed_path="NT_50_multi_seqlen=12k",
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        key="concat_avg_ws",
    ),
    "NT_100_multi": dict(
        embed_path="NT_100_multi_seqlen=12k",
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        key="concat_avg_ws",
    ),
    "NT_250_multi": dict(
        embed_path="NT_250_multi_seqlen=12k",
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        key="concat_avg_ws",
    ),
    "NT_500_multi": dict(
        embed_path="NT_500_multi_seqlen=12k",
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        key="concat_avg_ws",
    ),
    
    }

metrics = {
    "model_name": [],
    "bucket_id": [],
    "use_tissue": [],
    "C": [],
    "seed": [],
    "AUROC": [],
}

for model_name, downstream_kwargs in model_dict.items():
    print(f"********** Gathering results for: {model_name} **********")
    embed_path = downstream_kwargs["embed_path"]
    rc_aug = downstream_kwargs["rc_aug"]
    conjoin_train = downstream_kwargs["conjoin_train"]
    conjoin_test = downstream_kwargs["conjoin_test"]
    key = downstream_kwargs["key"]
    
    if "NT" in model_name: assert (rc_aug == False) and (conjoin_train == False) and (conjoin_test == False)
    
    base_embeds_path = PATH_TO_OUTPUTS
    embeds_path = osp.join(base_embeds_path, embed_path)
    
    print(f"Embed Path: {embeds_path}")
    train_val_ds_raw = torch.load(osp.join(embeds_path, "train_embeds.pt"),
                                    map_location="cpu")
    train_val_ds_raw = dataset_nan_filter(train_val_ds_raw, data_key=key)
    test_ds_raw = torch.load(osp.join(embeds_path, "test_embeds.pt"),
                             map_location="cpu")
    test_ds_raw = dataset_nan_filter(test_ds_raw, data_key=key)
    print(f"Total Train size: {len(train_val_ds_raw[key])},", end=" ")
    print(f"Total Test size: {len(test_ds_raw[key])},", end=" ")
    print(f"Shape: {test_ds_raw[key].shape[1:]}")

    for bucket_id, (min_dist, max_dist) in enumerate(DIST_TO_TSS):
        # Filter data to desired TSS bucket
        train_val_ds_filter = dataset_tss_filter(train_val_ds_raw, min_dist, max_dist)
        test_ds_filter = dataset_tss_filter(test_ds_raw, min_dist, max_dist)
        print(f"- TSS bucket: [{min_dist}, {max_dist}],", end=" ")
        print(f"Train size: {len(train_val_ds_filter[key])},", end=" ")
        print(f"Test size: {len(test_ds_filter[key])}")
    
        for use_tissue in USE_TISSUE:
            for C in Cs:
                for seed in range(1, 6):   
                    # Re-seed for SVM fitting
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                    svm_clf = make_pipeline(
                        StandardScaler(),
                        SVC(C=C, random_state=seed),
                    )

                    # Setup Train/Test dataset
                    if conjoin_train:
                        X = np.array(train_val_ds_filter[key])
                        X += np.array(train_val_ds_filter[f"rc_{key}"])
                        X /= 2
                    else:
                        X = np.array(train_val_ds_filter[key])
                    X_with_tissue = np.concatenate(
                        [X, np.array(train_val_ds_filter["tissue_embed"])[..., None]],
                        axis=-1
                    )
                    y = train_val_ds_filter["labels"]
                    if conjoin_train or conjoin_test:
                        X_test = np.array(test_ds_filter[key])
                        X_test += np.array(test_ds_filter[f"rc_{key}"])
                        X_test /= 2
                    else:
                        X_test = np.array(test_ds_filter[key])
                    X_test_with_tissue = np.concatenate(
                        [X_test, np.array(test_ds_filter["tissue_embed"])[..., None]],
                        axis=-1
                    )
                    y_test = test_ds_filter["labels"]

                    print(f"\tFitting SVM ({use_tissue=}, {C=}, {seed=})...", end=" ")
                    
                    mask = np.random.choice(len(X), size=5000, replace= 5000 > len(X) )
                    if use_tissue: 
                        X_train = X_with_tissue[mask]
                        X_test = X_test_with_tissue
                    else: 
                        X_train = X[mask]
                    y_train = y[mask]

                    start = time.time()
                    svm_clf.fit(X_train, y_train)
                    svm_y_pred = svm_clf.predict(X_test)
                    svm_aucroc = roc_auc_score(y_test, svm_y_pred)
                    end = time.time()
                    print(f"Completed! ({end - start:0.3f} s) -", end=" ")
                    print(f"AUROC: {svm_aucroc}")
                     
                    metrics["model_name"] += [model_name]
                    metrics["bucket_id"] += [bucket_id]
                    metrics["use_tissue"] += [use_tissue]
                    metrics["C"] += [C]
                    metrics["seed"] += [seed]
                    metrics["AUROC"] += [svm_aucroc]


df_metrics = pd.DataFrame.from_dict(metrics)
df_metrics.to_csv(osp.join(PATH_TO_OUTPUTS, "SVM_results.csv"))
