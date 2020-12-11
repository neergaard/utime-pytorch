import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm


def transition_matrix(y, n_classes=5, eval_frequency=[30]):

    # Generate count matrix
    M = np.zeros((n_classes, n_classes))
    for i, j in zip(y, y[1:]):
        M[i, j] += 1

    # Convert to probabilites
    M /= M.sum(axis=1, keepdims=True)

    return M


# def get_transitions_matrices(record_predictions, eval_frequency=[30]):


# fmt: off
def evaluate_performance(record_predictions, evaluation_windows=[1, 3, 5, 10, 15, 30], cases=['all', 'stable', 'transition']):
    """Evaluate the performance of the predicted results.

    Args:
        record_predictions (dict): dict containing predicted and true labels for every record
    """
    records = [r for r in record_predictions.keys()]
    ids = [r.split(".")[0] for r in records]
    df_total = []
    confmat_subject = {record: {eval_window: {case: None for case in cases} for eval_window in evaluation_windows} for record in records}
    confmat_total = {eval_window: {case: np.zeros((5, 5)) for case in cases} for eval_window in evaluation_windows}
    for eval_window in evaluation_windows:
        for case in cases:
            df = pd.DataFrame()
            df["FileID"] = records
            df["SubjectID"] = ids
            df["Window"] = f"{eval_window} s"
            df["Case"] = case
            print('')
            print(f'Evaluation window: {eval_window} | Case: {case}')
            for idx, record in enumerate(tqdm(records)):
                not_unknown_stage = record_predictions[record]['true'].sum(axis=1) == 1

                if case == 'all':
                    extract = np.full(record_predictions[record]['true'].shape[0], True) & not_unknown_stage
                elif case == 'stable':
                    extract = record_predictions[record]['stable_sleep'] & not_unknown_stage
                elif case == 'transition':
                    not_unknown_stage = record_predictions[record]['true'].sum(axis=1) == 1
                    extract = np.invert(record_predictions[record]['stable_sleep']) & not_unknown_stage

                # Get the true and predicted stages
                t = record_predictions[record]["true"][extract, :].argmax(axis=1)[::eval_window]
                p = np.mean(record_predictions[record]["predicted"][extract, :].reshape(-1, eval_window, 5), axis=1).argmax(axis=1)

                # Extract the metrics
                acc = accuracy_score(t, p)
                bal_acc = balanced_accuracy_score(t, p)
                kappa = cohen_kappa_score(t, p)
                f1 = f1_score(t, p, average="macro")
                prec = precision_score(t, p, average="macro")
                recall = recall_score(t, p, average="macro")
                mcc = matthews_corrcoef(t, p)

                # Assign metrics to dataframe
                df.loc[idx, "Accuracy"] = acc
                df.loc[idx, "Balanced accuracy"] = bal_acc
                df.loc[idx, "Kappa"] = kappa
                df.loc[idx, "F1"] = f1
                df.loc[idx, "Precision"] = prec
                df.loc[idx, "Recall"] = recall
                df.loc[idx, "MCC"] = mcc

                # Get stage-specific metrics
                precision, recall, f1, support = precision_recall_fscore_support(t, p, labels=[0, 1, 2, 3, 4])

                # Assign to dataframe
                for stage_idx, stage in zip([0, 1, 2, 3, 4], ["W", "N1", "N2", "N3", "REM"]):
                    df.loc[idx, f"F1 - {stage}"] = f1[stage_idx]
                    df.loc[idx, f"Precision - {stage}"] = precision[stage_idx]
                    df.loc[idx, f"Recall - {stage}"] = recall[stage_idx]
                    df.loc[idx, f"Support - {stage}"] = support[stage_idx]

                # Get confusion matrix
                C = confusion_matrix(t, p, labels=[0, 1, 2, 3, 4])
                confmat_subject[record][eval_window][case] = C
                confmat_total[eval_window][case] += C

            # Update list
            df_total.append(df)

    # Finalize dataframe
    df_total = pd.concat(df_total)

    return df_total, confmat_subject, confmat_total
    # fmt: on

    # metrics = {
    #     "record": records,
    #     "id": ids,
    #     "macro_f1": [],
    #     "micro_f1": [],
    #     "accuracy": [],
    #     "balanced_accuracy": [],
    #     "kappa": [],
    #     "mcc": [],
    #     "macro_recall": [],
    #     "micro_recall": [],
    #     "macro_precision": [],
    #     "micro_precision": [],
    # }
    # for record in records:

    #     y_true = record_predictions[record]["true_label"]
    #     y_pred = record_predictions[record]["predicted_label"]
    #     # labels = [0, 1, 2, 3, 4]

    #     metrics["macro_f1"].append(f1_score(y_true, y_pred, average="macro"))
    #     metrics["micro_f1"].append(f1_score(y_true, y_pred, average="micro"))
    #     metrics["accuracy"].append(accuracy_score(y_true, y_pred))
    #     metrics["balanced_accuracy"].append(balanced_accuracy_score(y_true, y_pred))
    #     metrics["kappa"].append(cohen_kappa_score(y_true, y_pred))
    #     metrics["mcc"].append(matthews_corrcoef(y_true, y_pred))
    #     metrics["macro_recall"].append(recall_score(y_true, y_pred, average="macro"))
    #     metrics["micro_recall"].append(recall_score(y_true, y_pred, average="micro"))
    #     metrics["macro_precision"].append(precision_score(y_true, y_pred, average="macro"))
    #     metrics["micro_precision"].append(precision_score(y_true, y_pred, average="micro"))
    #     # metrics["macro_f1"].append(f1_score(y_true, y_pred, labels=labels, average="macro"))
    #     # metrics["micro_f1"].append(f1_score(y_true, y_pred, labels=labels, average="micro"))
    #     # metrics["accuracy"].append(accuracy_score(y_true, y_pred))
    #     # metrics["balanced_accuracy"].append(balanced_accuracy_score(y_true, y_pred))
    #     # metrics["kappa"].append(cohen_kappa_score(y_true, y_pred, labels=labels))
    #     # metrics["mcc"].append(matthews_corrcoef(y_true, y_pred))
    #     # metrics["macro_recall"].append(recall_score(y_true, y_pred, labels=labels, average="macro"))
    #     # metrics["micro_recall"].append(recall_score(y_true, y_pred, labels=labels, average="micro"))
    #     # metrics["macro_precision"].append(precision_score(y_true, y_pred, labels=labels, average="macro"))
    #     # metrics["micro_precision"].append(precision_score(y_true, y_pred, labels=labels, average="micro"))
    # total_acc = []

    # return pd.DataFrame.from_dict(metrics).set_index("record")
