from glob import glob

import pandas as pd

from sava_ml_toolbox.utils.evaluation import SAVAMetrics

if __name__ == "__main__":

    root_path = "/path/to/your/dataset"
    ds_and_classes = glob(f"{root_path}/*/*")
    # create an empty pandas dataframe with the following columns: dataset, classes, mAP, precision, recall, f1
    df = pd.DataFrame(
        columns=["dataset", "classes", "mAP", "precision", "recall", "f1"]
    )
    for elem in ds_and_classes:
        dataset = elem.split("/")[-2]
        classes = elem.split("/")[-1]

        print(f"Dataset: {dataset} - Classes: {classes}")
        # if dataset == "isui":
        #     if classes == "Car":
        gt_json_path = f"{elem}/ground_truths.json"
        pred_json_path = f"{elem}/predictions.json"

        sava_metrics = SAVAMetrics(iou_type="segm")
        sava_metrics.load_gt(gt_json_path)
        sava_metrics.load_prediction(pred_json_path)

        sava_metrics.create_summary(f"{elem}/confusion_matrix.png")
        mAP = sava_metrics.mAP()
        precision = sava_metrics.precision()
        recall = sava_metrics.recall()
        f1 = sava_metrics.F1_score()

        new_row = pd.DataFrame(
            {
                "dataset": [dataset],
                "classes": [classes],
                "mAP": [f"{mAP:.3f}"],
                "precision": [f"{precision:.3f}"],
                "recall": [f"{recall:.3f}"],
                "f1": [f"{f1:.3f}"],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)
    # save df to csv
    df.to_csv("evaluation_results.csv", index=False)
