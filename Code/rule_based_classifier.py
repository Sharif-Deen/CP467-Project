"""
Rule-based (non machine learning) classifier for diabetic retinopathy severity.

Input:  Features/features.csv  (made by feature_extraction.py)
Output: Features/splits.csv           - shared 70/15/15 train/val/test split (use this for the ML model too)
        Features/rule_thresholds.json - the four severity cutoffs picked on the training split
        Features/rule_predictions.csv - predicted grade (0-4) for every image

How the rules work
------------------
1. Red lesion score.
   Microaneurysms and hemorrhages show up as small dark spots after the black-hat
   transform (dark_small_spots_10k). Sharp, bright images also have more vessel
   edges, which produce extra dark spots that are not lesions. So the spot count
   is divided by the edge density of the image, giving "dark spots per unit of
   vessel/edge structure". Log is taken so a few extreme images do not dominate.

       red_score = log(1 + dark_small_spots_10k / edge_density)

2. Thresholding into grades.
   More red lesions = more severe disease, so the score is cut into 5 bands:

       red_score <  t1        -> 0  No DR
       t1 <= red_score < t2   -> 1  Mild
       t2 <= red_score < t3   -> 2  Moderate
       t3 <= red_score < t4   -> 3  Severe
       red_score >= t4        -> 4  Proliferative

   The cutoffs t1..t4 are chosen ONLY on the training split, by trying many
   candidate values and keeping the ones with the best quadratic weighted kappa
   (the official APTOS metric). Validation and test images are never used to
   pick thresholds.

3. Referral rule (binary task from the comparison plan).
   grade >= 2 (red_score >= t2) -> "refer" ; otherwise "no referral".
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split

GRADE_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
RANDOM_STATE = 42  # keep fixed so every group member gets the same split


# ---------------------------------------------------------------- splitting
def make_splits(df):
    """70/15/15 split, stratified so every split has the same mix of grades."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df["diagnosis"],
                                   random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["diagnosis"],
                                 random_state=RANDOM_STATE)
    split = pd.Series("train", index=df.index)
    split[val.index] = "val"
    split[test.index] = "test"
    return split


# ------------------------------------------------------------------- rules
def red_lesion_score(df):
    """Dark small spots normalised by edge density (see rule 1 above)."""
    return np.log1p(df["dark_small_spots_10k"] / df["edge_density"]).to_numpy()


def apply_rules(scores, thresholds):
    """Rule 2: band the score into grades 0-4 using the four cutoffs."""
    return np.digitize(scores, thresholds)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def tune_thresholds(scores, labels, rounds=4, candidates=60):
    """
    Pick the four cutoffs on the training split.
    Start from the class proportions, then repeatedly try moving one cutoff at a
    time to each candidate value, keeping any change that raises the kappa.
    """
    # starting guess: put cutoffs where the cumulative class proportions fall
    proportions = np.bincount(labels, minlength=5) / len(labels)
    thresholds = list(np.quantile(scores, np.cumsum(proportions)[:4]))
    candidate_values = np.quantile(scores, np.linspace(0.05, 0.995, candidates))

    best = qwk(labels, apply_rules(scores, thresholds))
    for _ in range(rounds):
        for i in range(4):
            for value in candidate_values:
                trial = sorted(thresholds[:i] + [value] + thresholds[i + 1:])
                score = qwk(labels, apply_rules(scores, trial))
                if score > best:
                    best, thresholds = score, trial
    return [float(t) for t in thresholds], best


# -------------------------------------------------------------- evaluation
def report(name, y_true, y_pred):
    """Print the metrics from the comparison plan for one split."""
    refer_true = y_true >= 2
    refer_pred = y_pred >= 2
    tp = np.sum(refer_true & refer_pred)
    tn = np.sum(~refer_true & ~refer_pred)
    sensitivity = tp / max(refer_true.sum(), 1)
    specificity = tn / max((~refer_true).sum(), 1)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "qwk": qwk(y_true, y_pred),
        "baseline_accuracy_always_0": float(np.mean(y_true == 0)),
        "referral_accuracy": float(np.mean(refer_true == refer_pred)),
        "referral_sensitivity": float(sensitivity),
        "referral_specificity": float(specificity),
    }

    print(f"\n===== {name} ({len(y_true)} images) =====")
    for key, value in results.items():
        print(f"{key:28s} {value:.3f}")
    print("Confusion matrix (rows = true grade 0-4, cols = predicted 0-4):")
    print(confusion_matrix(y_true, y_pred, labels=range(5)))
    return results


# -------------------------------------------------------------------- main
def main(features_csv, output_dir):
    df = pd.read_csv(features_csv)

    # reuse an existing split file so results stay comparable with the ML model
    split_csv = os.path.join(output_dir, "splits.csv")
    if os.path.exists(split_csv):
        split_map = pd.read_csv(split_csv).set_index("image_id")["split"]
        df["split"] = df["image_id"].map(split_map)
        if df["split"].isna().any():
            raise RuntimeError("splits.csv does not match features.csv; delete it and rerun")
    else:
        df["split"] = make_splits(df)
        df[["image_id", "diagnosis", "split"]].to_csv(split_csv, index=False)
        print("Saved shared split to", split_csv)

    df["red_score"] = red_lesion_score(df)
    train = df[df["split"] == "train"]

    thresholds, train_qwk = tune_thresholds(train["red_score"].to_numpy(),
                                            train["diagnosis"].to_numpy())
    print("Cutoffs t1..t4 (chosen on train):", np.round(thresholds, 3))

    with open(os.path.join(output_dir, "rule_thresholds.json"), "w") as f:
        json.dump({"feature": "log1p(dark_small_spots_10k / edge_density)",
                   "thresholds": thresholds}, f, indent=2)

    df["predicted"] = apply_rules(df["red_score"].to_numpy(), thresholds)
    df["predicted_name"] = [GRADE_NAMES[g] for g in df["predicted"]]

    all_results = {}
    for split_name in ["train", "val", "test"]:
        part = df[df["split"] == split_name]
        all_results[split_name] = report(split_name, part["diagnosis"].to_numpy(),
                                         part["predicted"].to_numpy())

    df[["image_id", "class_name", "diagnosis", "split", "red_score",
        "predicted", "predicted_name"]].to_csv(
        os.path.join(output_dir, "rule_predictions.csv"), index=False)

    with open(os.path.join(output_dir, "rule_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved predictions and results to", output_dir)


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    features_dir = os.path.join(project_dir, "Features")
    main(os.path.join(features_dir, "features.csv"), features_dir)
