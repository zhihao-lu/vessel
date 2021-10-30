import cv2.cv2 as cv2
import glob
import os
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier
import torch
from pathlib import Path
import json
import csv


def get_frames_from_json(path):
    """
    Converts given json of frames to tuple

    Args:
        path (str): Path of input JSON.

    Returns:
        tuple: Tuple of frame numbers.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    return tuple(data['frames_to_infer'])


def write_to_csv(rows, filename):
    """
    Write given list to appropriate .csv output

    Args:
        rows (list[dict]): List of dictionaries of data to write to csv
        filename (str): Filename without .csv extension

    Returns:
        None
    """
    headers = ["frame_index", "no_of_ships", "no_of_kayaks", "ships_coordinates", "kayaks_coordinates"]

    with open(f'{filename}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def tensor_to_yolo(x, conf, xmax=1920, ymax=1080):
    """
    Converts tensors to yolov3 coordinate format

    Args:
        x (list[tensor]): list of coordinates in tensor form.
        conf (float): Confidence score.
        xmax (int): Max value of x dimension (width).
            Defaults to 1920.
        ymax (int): Max value of y dimension (height).
            Defaults to 1080.

    Returns:
        list: List of string of coordinates in yoloV3 format to 6dp and confidence score
    """
    x1, y1, x2, y2 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    x = (x1 + x2)/2
    y = (y1 + y2)/2
    w = x2 - x1
    h = y2 - y1
    conf = round(conf, 2)
    return list(map(lambda x: str(round(x, 6)), (x/xmax, y/ymax, w/xmax, h/ymax, conf)))


def process(dataset, device, save_path, model, names, keep=(), skip=4):
    """
    Processes given video to generate .mp4 and .csv output

    Args:
        dataset (Dataset): Dataset object with file as input, generated using LoadImages function from utils.dataset.
        device (Device): Device object generated from select_device.
        save_path (str): Filename without extensions.
        model (Model): Model with weights loaded using attempt_load from models.experimental
        names (list): List of classes.
        keep (Tuple): Tuple of frame numbers to keep and output to .csv.
        skip (int): Number of frames to skip to improve efficiency.
            Defaults to 4

    Returns:
        None
    """
    bs = max(1, len(dataset))
    vid_path, vid_writer = [None] * bs, [None] * bs

    rows = []
    for idx, t in enumerate(dataset):
        v_count, k_count, v_coord, k_coord = 0, 0, [], []
        path, img, im0s, vid_cap = t
        row_d = {}
        if not (idx % skip) or idx in keep:
            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # normalize image

            if img.ndimension() == 3:
                img = img.unsqueeze(0)  # Include batch dimension

                pred = model(img)[0]

                # Apply NMS (https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)
                # To group multiple bounding boxes into 1 based on IOU
                pred = non_max_suppression(pred)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            if idx in keep:
                                if c == 0:
                                    v_count += 1
                                    v_coord.append(tensor_to_yolo(xyxy, conf))
                                else:
                                    k_count += 1
                                    k_coord.append(tensor_to_yolo(xyxy, conf))

                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)
                        if idx in keep:
                            row_d["no_of_ships"] = v_count
                            row_d["no_of_kayaks"] = k_count
                            row_d["ships_coordinates"] = ";".join(sorted(map(lambda x: "_".join(x), v_coord))) + ";" if v_coord else "-"
                            row_d["kayaks_coordinates"] = ";".join(sorted(map(lambda x: "_".join(x), k_coord))) + ";" if k_coord else "-"
                            row_d["frame_index"] = idx
                            rows.append(row_d)
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)// (skip/2)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path + "_processed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
    write_to_csv(rows, save_path)


def process_video(video_path, json_path, weights_path, d="cpu"):
    """
    Runs entire process.

    Args:
        video_path (str): Path to video file.
        json_path (str): Path to JSON file.
        weights_path (str): Path to weights file.
        d (str): Device to use.
            Defaults to 'cpu'.

    Returns
        None
    """
    start = time.time()

    # Prepare dataset
    dataset = LoadImages(video_path)
    device = select_device(d)
    frames = get_frames_from_json(json_path)

    # Prepare model
    model = attempt_load(weights_path, map_location="cpu")
    names = model.module.names if hasattr(model, 'module') else model.names

    # Generate output
    output_path = video_path[:-4]
    process(dataset, device, output_path, model, names, keep=frames)

    print(time.time() - start)
