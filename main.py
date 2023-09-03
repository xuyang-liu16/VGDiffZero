from collections import defaultdict
import json
import argparse
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from interpreter import *
from executor import *
from methods import *


METHODS_MAP = {
    "full_exp": Full_exp,
    "core_exp": Core_exp,
    "random": Random,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file with expressions and annotations in jsonlines format")
    parser.add_argument("--image_root", type=str, help="path to images (train2014 directory of COCO)")
    parser.add_argument("--diffusion_model", type=str, default="2-1", help="Stable Diffusion model version")
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--n_samples', nargs='+', type=int, default=[5,10,25])
    parser.add_argument("--method", type=str, default="full_exp", help="method to solve expressions")
    parser.add_argument("--box_representation_method", type=str, default="crop", help="method of representing boxes as individual images")
    parser.add_argument("--box_method_aggregator", type=str, default="sum", help="method of combining box representation scores")
    parser.add_argument("--box_area_threshold", type=float, default=0.0, help="minimum area (as a proportion of image area) for a box to be considered as the answer")
    parser.add_argument("--output_file", type=str, default=None, help="(optional) output path to save results")
    parser.add_argument("--detector_file", type=str, default=None, help="(optional) file containing object detections. if not provided, the gold object boxes will be used.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="number of instances to process in one model call (only supported for baseline model)")
    
    args = parser.parse_args()

    with open(args.input_file, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    executor = VGDiffZeroExecutor(version=args.diffusion_model, n_trials=args.n_trials, n_samples=args.n_samples, box_representation_method=args.box_representation_method, method_aggregator=args.box_method_aggregator, device=device)
    method = METHODS_MAP[args.method](args)
    correct_count = 0
    total_count = 0
    if args.output_file:
        output_file = open(args.output_file, "w")
    if args.detector_file:
        detector_file = open(args.detector_file)
        detections_list = json.load(detector_file)
        if isinstance(detections_list, dict):
            detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
        else:
            detections_map = defaultdict(list)
            for detection in detections_list:
                detections_map[detection["image_id"]].append(detection["box"])
    batch_count = 0
    batch_boxes = []
    batch_gold_boxes = []
    batch_gold_index = []
    batch_file_names = []
    batch_sentences = []
    for datum in tqdm.tqdm(data):
        if "coco" in datum["file_name"].lower():
            file_name = "_".join(datum["file_name"].split("_")[:-1])+".jpg"
        else:
            file_name = datum["file_name"]
        img_path = os.path.join(args.image_root, file_name)
        img = Image.open(img_path).convert('RGB')
        gold_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in datum["anns"]]
        if isinstance(datum["ann_id"], int) or isinstance(datum["ann_id"], str):
            datum["ann_id"] = [datum["ann_id"]]
        assert isinstance(datum["ann_id"], list)
        gold_index = [i for i in range(len(datum["anns"])) if datum["anns"][i]["id"] in datum["ann_id"]]
        for sentence in datum["sentences"]:
            if args.detector_file:
                boxes = [Box(x=box[0], y=box[1], w=box[2], h=box[3]) for box in detections_map[int(datum["image_id"])]]
                if len(boxes) == 0:
                    boxes = [Box(x=0, y=0, w=img.width, h=img.height)]
            else:
                boxes = gold_boxes
            env = Environment(img, boxes, executor, str(datum["image_id"]))
            result = method.execute(sentence["raw"].lower(), env)
            boxes = env.boxes
            print(sentence["raw"].lower())
            correct = False
            for g_index in gold_index:
                if iou(boxes[result["pred"]], gold_boxes[g_index]) > 0.5:
                    correct = True
                    break
            if correct:
                result["correct"] = 1
                correct_count += 1
            else:
                result["correct"] = 0
            if args.detector_file:
                argmax_ious = []
                max_ious = []
                for g_index in gold_index:
                    ious = [iou(box, gold_boxes[g_index]) for box in boxes]
                    argmax_iou = -1
                    max_iou = 0
                    if max(ious) >= 0.5:
                        for index, value in enumerate(ious):
                            if value > max_iou:
                                max_iou = value
                                argmax_iou = index
                    argmax_ious.append(argmax_iou)
                    max_ious.append(max_iou)
                argmax_iou = -1
                max_iou = 0
                if max(max_ious) >= 0.5:
                    for index, value in zip(argmax_ious, max_ious):
                        if value > max_iou:
                            max_iou = value
                            argmax_iou = index
                result["gold_index"] = argmax_iou
            else:
                result["gold_index"] = gold_index
            result["bboxes"] = [[box.left, box.top, box.right, box.bottom] for box in boxes]
            result["file_name"] = file_name
            result["probabilities"] = result["probs"]
            result["text"] = sentence["raw"].lower()
            if args.output_file:
                # Serialize numpy arrays for JSON.
                for key in result:
                    if isinstance(result[key], np.ndarray):
                        result[key] = result[key].tolist()
                    if isinstance(result[key], np.int64):
                        result[key] = result[key].item()
                output_file.write(json.dumps(result)+"\n")
            total_count += 1
            print(f"est_acc: {100 * correct_count / total_count:.3f}")

    if args.output_file:
        output_file.close()
    print(f"acc: {100 * correct_count / total_count:.3f}")
    stats = method.get_stats()
    if stats:
        pairs = sorted(list(stats.items()), key=lambda tup: tup[0])
        for key, value in pairs:
            if isinstance(value, float):
                print(f"{key}: {value:.5f}")
            else:
                print(f"{key}: {value}")
