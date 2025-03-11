import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import termplotlib as tpl
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from sava_ml_toolbox.utils.utils import NoVerbose


class SAVAMetrics:
    def __init__(self, iou_type="segm"):
        """
        Initialize the COCOMetrics class with a categories dictionary.

        Args:
            categories (dict): Dictionary mapping category IDs to category names.
        """
        self.coco_gt = None
        self.coco_dt = None
        self.coco_eval = None
        self.iou_type = iou_type

    def load_prediction(self, pred_json_path):
        """
        Load the predictions from a JSON file.

        Args:
            pred_json_path (str): Path to the predictions JSON file.
        """
        with open(pred_json_path, "r") as f:
            predictions = json.load(f)
        with NoVerbose():
            self.coco_dt = self.coco_gt.loadRes(predictions)

    def load_gt(self, gt_json_path):
        """
        Load the ground truth from a JSON file.

        Args:
            gt_json_path (str): Path to the ground truth JSON file.
        """
        with NoVerbose():
            self.coco_gt = COCO(gt_json_path)

    def COCOeval(self):
        """
        Run COCO evaluation using pycocotools.

        Returns:
            COCOeval: The COCO evaluation object.
        """
        if self.coco_gt is None or self.coco_dt is None:
            raise ValueError(
                "Ground truth and predictions must be loaded before running COCO evaluation."
            )

        # with NoVerbose():
        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, self.iou_type)
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        return self.coco_eval

    def F1_score(self):
        """
        Calculate the F1 score using pycocotools.

        Returns:
            float: The F1 score.
        """
        if self.coco_eval is None:
            self.COCOeval()

        # Calculate F1 score from precision and recall
        precision = self.coco_eval.stats[1]  # Precision at IoU=0.50:0.95
        recall = self.coco_eval.stats[8]  # Recall at IoU=0.50:0.95
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def precision(self):
        """
        Calculate the precision using pycocotools.

        Returns:
            float: The precision score.
        """
        if self.coco_eval is None:
            self.COCOeval()

        return self.coco_eval.stats[1]

    def recall(self):
        """
        Calculate the recall using pycocotools.

        Returns:
            float: The recall score.
        """
        if self.coco_eval is None:
            self.COCOeval()

        return self.coco_eval.stats[8]

    def mAP(self):
        """
        Calculate the mean Average Precision (mAP) using pycocotools.

        Returns:
            float: The mAP score.
        """

        # mAP is the first element in the stats array
        return self.coco_eval.stats[0]

    def singular_confusion_matrix(self):
        """
        Get the aggregated confusion matrix according to the categories from the COCO evaluation.

        Returns:
            dict: A dictionary representing the confusion matrix.
        """
        if self.coco_eval is None:
            self.COCOeval()

        confusion_matrix = defaultdict(lambda: defaultdict(int))
        category_ids = self.coco_gt.getCatIds()

        for eval_img in self.coco_eval.evalImgs:
            if eval_img is None:
                continue
            gt_ids = eval_img["gtIds"]
            dt_ids = eval_img["dtIds"]
            gt_matches = eval_img["gtMatches"]
            dt_matches = eval_img["dtMatches"]
            gt_cats = [self.coco_gt.anns[gt_id]["category_id"] for gt_id in gt_ids]
            dt_cats = [self.coco_dt.anns[dt_id]["category_id"] for dt_id in dt_ids]

            for gt_id, gt_match, gt_cat in zip(gt_ids, gt_matches[0], gt_cats):
                if gt_match == 0:
                    confusion_matrix[gt_cat]["FN"] += 1
                else:
                    confusion_matrix[gt_cat]["TP"] += 1

            for dt_id, dt_match, dt_cat in zip(dt_ids, dt_matches[0], dt_cats):
                if dt_match == 0:
                    confusion_matrix[dt_cat]["FP"] += 1

            # Calculate TN for each category
            for cat_id in category_ids:
                if cat_id not in gt_cats and cat_id not in dt_cats:
                    confusion_matrix[cat_id]["TN"] += 1

        return confusion_matrix

    def confusion_matrix(self):
        """
        Get the confusion matrix comparing predicted categories against ground truth categories.

        Returns:
            pd.DataFrame: A DataFrame representing the confusion matrix.
        """
        if self.coco_eval is None:
            self.COCOeval()

        category_ids = self.coco_gt.getCatIds()
        category_names = [cat["name"] for cat in self.coco_gt.loadCats(category_ids)]
        confusion_matrix = np.zeros((len(category_ids), len(category_ids)), dtype=int)

        for eval_img in self.coco_eval.evalImgs:
            if eval_img is None:
                continue
            gt_ids = eval_img["gtIds"]
            dt_ids = eval_img["dtIds"]
            gt_matches = eval_img["gtMatches"]
            dt_matches = eval_img["dtMatches"]
            gt_cats = [self.coco_gt.anns[gt_id]["category_id"] for gt_id in gt_ids]
            dt_cats = [self.coco_dt.anns[dt_id]["category_id"] for dt_id in dt_ids]

            for gt_cat, dt_cat, gt_match in zip(gt_cats, dt_cats, gt_matches[0]):
                gt_index = category_ids.index(gt_cat)
                if gt_match == 0:
                    # False negative: ground truth category but no matching detection
                    confusion_matrix[gt_index, -1] += 1
                else:
                    dt_index = category_ids.index(dt_cat)
                    confusion_matrix[gt_index, dt_index] += 1

        # Normalize the confusion matrix by row (ground truth categories)
        confusion_matrix = (
            confusion_matrix.astype("float")
            / confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        df_cm = pd.DataFrame(
            confusion_matrix, index=category_names, columns=category_names
        )
        return df_cm

    def plot_confusion_matrix(self, df_cm, output_path="confusion_matrix.png"):
        """
        Plot the confusion matrix and save it as an image.

        Args:
            df_cm (pd.DataFrame): The confusion matrix to plot.
            output_path (str): The path to save the confusion matrix image.
        """
        plt.figure()
        sns.heatmap(
            df_cm, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Ratio"}
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Ground Truth Categories")
        plt.xlabel("Predicted Categories")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def scale_dimensions(self, annotation, out_size=640):
        ann_w = self.coco_gt.imgs[annotation["image_id"]]["width"]
        ann_h = self.coco_gt.imgs[annotation["image_id"]]["height"]

        return (annotation["area"] / (ann_w * ann_h)) * (out_size**2)

    def calculate_metrics_by_area(self):
        """
        Calculate the F1 score and mAP for all areas, dividing the areas into a histogram with 100 bins.

        Returns:
            dict: A dictionary with the F1 score and mAP for each bin.
        """
        if self.coco_gt is None or self.coco_dt is None:
            raise ValueError(
                "Ground truth and predictions must be loaded before running COCO evaluation."
            )

        # Extract areas from ground truth annotations
        areas = [
            self.scale_dimensions(ann)
            for ann in self.coco_gt.loadAnns(self.coco_gt.getAnnIds())
        ]

        bins = [10**i for i in range(0, 7)]

        # Draw histogram in the terminal
        hist, bin_edges = np.histogram(areas, bins=bins)
        fig = tpl.figure()
        fig.hist(hist, bin_edges, orientation="horizontal", force_ascii=False)
        fig.show()
        print()

        metrics_by_area = {}

        for i in tqdm(range(len(bins) - 1)):
            with NoVerbose():
                min_bin = bins[i]
                max_bin = bins[i + 1]

                # Filter annotations and predictions for the current bin
                reduced_gt_ids = [
                    ann
                    for ann in self.coco_gt.anns
                    if min_bin
                    <= self.scale_dimensions(self.coco_gt.anns[ann])
                    < max_bin
                ]

                bin_gt = self.coco_gt.loadAnns(reduced_gt_ids)

                if len(bin_gt) == 0:
                    metrics_by_area[f"area_{min_bin}_{max_bin}"] = {
                        "F1_score": "gt_missing",
                        "mAP": "gt_missing",
                    }
                    continue

                reduced_dt_ids = [
                    ann
                    for ann in self.coco_dt.anns
                    if min_bin
                    <= self.scale_dimensions(self.coco_dt.anns[ann])
                    < max_bin
                ]
                bin_dt = self.coco_dt.loadAnns(reduced_dt_ids)

                if len(bin_dt) == 0:
                    metrics_by_area[f"area_{min_bin}_{max_bin}"] = {
                        "F1_score": "det_missing",
                        "mAP": "det_missing",
                    }
                    continue

                # Create temporary COCO objects for the current bin
                temp_coco_gt = COCO()
                # temp_coco_gt.dataset["annotations"] = bin_gt
                temp_coco_gt.dataset = {
                    "images": self.coco_gt.dataset["images"],
                    "annotations": bin_gt,
                    "categories": self.coco_gt.dataset["categories"],
                }
                temp_coco_gt.createIndex()

                temp_coco_dt = self.coco_gt.loadRes(
                    bin_dt
                )  # TODO Fix the case in which the dt is empty

                # Run COCO evaluation for the current bin
                coco_eval = COCOeval(temp_coco_gt, temp_coco_dt, self.iou_type)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                if len(coco_eval.stats) > 0:
                    # Calculate F1 score and mAP
                    precision = coco_eval.stats[1]  # Precision at IoU=0.50:0.95
                    recall = coco_eval.stats[8]  # Recall at IoU=0.50:0.95
                    if precision + recall == 0:
                        f1_score = 0.0
                    else:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                    mAP = coco_eval.stats[
                        0
                    ]  # mAP is the first element in the stats array
                else:
                    f1_score = 0.0
                    mAP = 0.0

            metrics_by_area[f"area_{min_bin}_{max_bin}"] = {
                "F1_score": f1_score,
                "mAP": mAP,
            }

        return metrics_by_area

    def calculate_metrics_by_image_center_distance(self, num_bins=5):

        if self.coco_gt is None or self.coco_dt is None:
            raise ValueError(
                "Ground truth and predictions must be loaded before running COCO evaluation."
            )
        for id in self.coco_gt.anns:
            ann = self.coco_gt.anns[id]
            img_id = ann["image_id"]
            img_width = self.coco_gt.imgs[img_id]["width"]
            img_height = self.coco_gt.imgs[img_id]["height"]
            bbox_center = [
                ann["bbox"][0] + (ann["bbox"][2] / 2),
                ann["bbox"][1] + (ann["bbox"][3] / 2),
            ]
            distance = np.sqrt(
                ((img_width / 2) - bbox_center[0]) ** 2
                + ((img_height / 2) - bbox_center[1]) ** 2
            )
            ann["dist_from_center"] = distance / (
                (np.sqrt(img_width**2 + img_height**2)) / 2
            )

        for id in self.coco_dt.anns:
            ann = self.coco_dt.anns[id]
            img_id = ann["image_id"]
            img_width = self.coco_gt.imgs[img_id]["width"]
            img_height = self.coco_gt.imgs[img_id]["height"]
            bbox_center = [
                ann["bbox"][0] + ann["bbox"][2] / 2,
                ann["bbox"][1] + ann["bbox"][3] / 2,
            ]
            distance = np.sqrt(
                ((img_width / 2) - bbox_center[0]) ** 2
                + ((img_height / 2) - bbox_center[1]) ** 2
            )
            ann["dist_from_center"] = distance / (
                (np.sqrt(img_width**2 + img_height**2)) / 2
            )

        distances = np.array(
            [
                ann["dist_from_center"]
                for ann in self.coco_gt.loadAnns(self.coco_gt.getAnnIds())
            ]
        )
        # bins = np.linspace(0, max(distances), num_bins + 1)
        bins = np.linspace(0, 1, num_bins + 1)

        # Draw histogram in the terminal
        hist, bin_edges = np.histogram(distances, bins=bins)
        fig = tpl.figure()
        fig.hist(hist, bin_edges, orientation="horizontal", force_ascii=False)
        fig.show()
        # Extract areas from ground truth annotations
        print()

        metrics_by_distance = {}

        for i in tqdm(range(len(bins) - 1)):
            with NoVerbose():
                min_bin = bins[i]
                max_bin = bins[i + 1]

                # Filter annotations and predictions for the current bin
                reduced_gt_ids = [
                    ann
                    for ann in self.coco_gt.anns
                    if min_bin <= self.coco_gt.anns[ann]["dist_from_center"] < max_bin
                ]
                bin_gt = self.coco_gt.loadAnns(reduced_gt_ids)

                if len(bin_gt) == 0:
                    metrics_by_distance[f"center_{min_bin}_{max_bin}"] = {
                        "F1_score": "gt_missing",
                        "mAP": "gt_missing",
                    }
                    continue

                reduced_dt_ids = [
                    ann
                    for ann in self.coco_dt.anns
                    if min_bin <= self.coco_dt.anns[ann]["dist_from_center"] < max_bin
                ]
                bin_dt = self.coco_dt.loadAnns(reduced_dt_ids)

                if len(bin_dt) == 0:
                    metrics_by_distance[f"center_{min_bin}_{max_bin}"] = {
                        "F1_score": "det_missing",
                        "mAP": "det_missing",
                    }
                    continue

                # Create temporary COCO objects for the current bin
                temp_coco_gt = COCO()
                # temp_coco_gt.dataset["annotations"] = bin_gt
                temp_coco_gt.dataset = {
                    "images": self.coco_gt.dataset["images"],
                    "annotations": bin_gt,
                    "categories": self.coco_gt.dataset["categories"],
                }
                temp_coco_gt.createIndex()

                temp_coco_dt = self.coco_gt.loadRes(bin_dt)

                # Run COCO evaluation for the current bin
                coco_eval = COCOeval(temp_coco_gt, temp_coco_dt, self.iou_type)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # if len(coco_eval.stats) > 0:
                # Calculate F1 score and mAP
                precision = coco_eval.stats[1]  # Precision at IoU=0.50:0.95
                recall = coco_eval.stats[8]  # Recall at IoU=0.50:0.95
                if precision + recall == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                mAP = coco_eval.stats[0]  # mAP is the first element in the stats array
                # else:
                #     f1_score = 0.0
                #     mAP = 0.0

            metrics_by_distance[f"center_{min_bin}_{max_bin}"] = {
                "F1_score": f1_score,
                "mAP": mAP,
            }

        return metrics_by_distance

    def create_summary(self, cm_output_path="confusion_matrix.png"):
        """
        Create a summary of the evaluation metrics, including the F1 score, mAP, metrics by area, and metrics by distance.
        Also, plot the confusion matrix.

        Returns:
            None
        """

        print()
        print(f"{'=' * 20} Overall Metrics {'=' * 20}")
        print()
        if self.coco_eval is None:
            self.COCOeval()

        # Overall metrics
        f1_score = self.F1_score()
        map_score = self.mAP()

        # Print the results
        print()
        print(f"{'Metric':<20}{'Value':<10}")
        print(f"{'F1 Score':<20}{f1_score:.4f}")
        print(f"{'mAP':<20}{map_score:.4f}")
        print()

        # Metrics by area
        print(f"{'=' * 20} Metrics by Area {'=' * 20}")
        print()
        metrics_by_area = self.calculate_metrics_by_area()
        print()
        print(f"{'Area Bin':<30}{'F1 Score':<20}{'mAP':<20}")
        for bin_name, metrics in metrics_by_area.items():
            title = (
                "area "
                + str(int(float(bin_name.split("_")[1])))
                + " to "
                + str(int(float(bin_name.split("_")[2])))
            )
            if metrics["F1_score"] == "gt_missing":
                print(f"{title:<30}{'GT missing':<20}{'GT missing':<20}")
            elif metrics["F1_score"] == "det_missing":
                print(f"{title:<30}{'Det missing':<20}{'Det missing':<20}")
            else:
                print(
                    f"{title:<30}{float(metrics['F1_score']):<20.4f}{float(metrics['mAP']):<20.4f}"
                )
        print()

        # Metrics by distance
        print(f"{'=' * 20} Metrics by Distance {'=' * 20}")
        print()
        metrics_by_distance = self.calculate_metrics_by_image_center_distance()
        print()
        print(f"{'Distance':<30}{'F1 Score':<20}{'mAP':<20}")
        for distance, metrics in metrics_by_distance.items():
            title = (
                "dist "
                + str(round(float(distance.split("_")[1]), 1))
                + " to "
                + str(round(float(distance.split("_")[2]), 1))
            )
            if metrics["F1_score"] == "gt_missing":
                print(f"{title:<30}{'GT missing':<20}{'GT missing':<20}")
            elif metrics["F1_score"] == "det_missing":
                print(f"{title:<30}{'Det missing':<20}{'Det missing':<20}")
            else:
                print(
                    f"{title:<30}{float(metrics['F1_score']):<20.4f}{float(metrics['mAP']):<20.4f}"
                )
        print()

        print(f"{'=' * 20} Confusion Matrix {'=' * 20}")
        print()

        # Confusion matrix
        df_cm = self.confusion_matrix()
        self.plot_confusion_matrix(df_cm, output_path=cm_output_path)
        print(f"Confusion matrix saved in {cm_output_path}")
