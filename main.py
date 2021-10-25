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

WEIGHTS = os.path.join("exp86", "weights", "best.pt")



RESULTS_FOLDER = "pred"

model = attempt_load(WEIGHTS, map_location="cpu")
names = model.module.names if hasattr(model, 'module') else model.names


def get_frames_from_json(path):
    with open(path) as json_file:
        data = json.load(json_file)

    return tuple(data['frames_to_infer'])


def write_to_csv(rows, filename="a"):
    headers = ["frame_index", "no_of_ships", "no_of_kayaks", "ships_coordinates", "kayaks_coordinates"]

    with open(f'{filename}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def tensor_to_yolo(x, xmax=1920, ymax=1080):
    x1, y1, x2, y2 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    x = (x1 + x2)/2
    y = (y1 + y2)/2
    w = x2 - x1
    h = y2 - y1
    return list(map(lambda x: str(round(x, 6)), (x/xmax, y/ymax, w/xmax, h/ymax)))


dataset = LoadImages("horizon_1_ship.avi")
device = select_device("cpu")


def d(dataset, device, save_path, keep = (), skip=4):
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
                                    v_coord.append(tensor_to_yolo(xyxy))
                                else:
                                    k_count += 1
                                    k_coord.append(tensor_to_yolo(xyxy))

                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)
                        if idx in keep:
                            row_d["no_of_ships"] = v_count
                            row_d["no_of_kayaks"] = k_count
                            row_d["ships_coordinates"] = ";".join(sorted(map(lambda x: "_".join(x), v_coord))) + ";" if v_coord else "-"
                            row_d["kayaks_coordinates"] = ";".join(sorted(map(lambda x: "_".join(x), k_coord))) + ";" if k_coord else "-"
                            row_d["frame_index"] = idx
                            rows.append(row_d)
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)// (skip/2)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
    write_to_csv(rows)

import time
def test():
    start = time.time()
    d(dataset, device, "pred/a.mp4", keep = [0,12,32,44])
    print(time.time() - start)


