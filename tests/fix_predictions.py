import json
from glob import glob

from sava_ml_toolbox.utils.evaluation import SAVAMetrics

if __name__ == "__main__":

    root_path = "/home/fabmo/works/sava-ml-toolbox/data/datasets"
    ds_and_classes = glob(f"{root_path}/*/*")
    # for elem in ds_and_classes:

    #     pred_json_path = f"{elem}/predictions.json"
    #     print(f"Processing {pred_json_path}")
    #     # load predictions json file
    #     with open(pred_json_path, "r") as f:
    #         data = json.load(f)

    #     # go through all the annotations and remove the ones where the segmentation mask is empty
    #     for i in range(len(data) - 1, -1, -1):
    #         if data[i]["segmentation"] == [] or data[i]["segmentation"] == [[]]:
    #             print(f"Removing annotation {data[i]['image_id']} from the predictions")
    #             data.pop(i)
    #         if len(data[i]["segmentation"][0]) <= 4:
    #             print(
    #                 f"Removing bbox annotation {data[i]['image_id']} from the predictions"
    #             )
    #             data.pop(i)

    #     # save the updated predictions json file
    #     with open(pred_json_path, "w") as f:
    #         json.dump(data, f)

    for elem in ds_and_classes:
        pred_json_path = f"{elem}/ground_truths.json"
        print(f"Processing {pred_json_path}")

        # load predictions json file
        with open(pred_json_path, "r") as f:
            data = json.load(f)

        # go through all the annotations and remove the ones where the segmentation mask is empty
        for i in range(len(data["annotations"]) - 1, -1, -1):
            if data["annotations"][i]["segmentation"] == [] or data["annotations"][i][
                "segmentation"
            ] == [[]]:
                print(
                    f"Removing annotation {data['annotations'][i]['image_id']} from the predictions"
                )
                data["annotations"].pop(i)

        # save the updated predictions json file
        with open(pred_json_path, "w") as f:
            json.dump(data, f)
