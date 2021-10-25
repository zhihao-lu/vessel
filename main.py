import cv2.cv2 as cv2
import glob
import pandas as pd
import os
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier
import torch
from pathlib import Path

WEIGHTS = os.path.join("exp86", "weights", "best.pt")


def write_video(directory, f, fps=12, codec="XVID"):
    img_array = []
    for filename in glob.glob(directory + '/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(f, cv2.VideoWriter_fourcc(*codec), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()


RESULTS_FOLDER = "pred"

model = attempt_load(WEIGHTS, map_location="cpu")
names = model.module.names if hasattr(model, 'module') else model.names

def p(frame, model, output_dir, num):
    ori = frame.copy()
    device = select_device("cpu")
    frame = cv2.resize(frame, (384, 640))
    frame = torch.from_numpy(frame.T).to(device).float()
    frame /= 255.0
    frame = frame.unsqueeze(0)

    im0_shape = (1080, 1920, 3)
    pred = model(frame)[0]


    # Apply NMS (https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)
    # To group multiple bounding boxes into 1 based on IOU
    pred = non_max_suppression(pred)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], im0_shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                plot_one_box(xyxy, ori, label=label, color=colors(c, True), line_thickness=3)

        cv2.imwrite(output_dir + '/' + str(num).zfill(5) + '_test.jpg', ori)
    return pred


def get_frames(path, output_dir, keep=(), skip=2):
    vid = cv2.VideoCapture(path)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        ret, frame = vid.read()



        if not (i % skip) or i in keep:
            #p(frame, model, "pred", i)
            cv2.imwrite(output_dir + '/' + str(i).zfill(5) + '_test.jpg', frame)

    vid.release()
    cv2.destroyAllWindows()

dataset = LoadImages("horizon_1_ship.avi")
device = select_device("cpu")
def d(dataset, device, keep = (), skip=4):
    a = 0
    bs = max(1, len(dataset))
    vid_path, vid_writer = [None] * bs, [None] * bs
    for idx, t in enumerate(dataset):
        path, img, im0s, vid_cap = t
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

                    p = Path(p)  # to Path
                    save_path = f"pred/a.mp4"
                    a += 1# img.jpg
                    # s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        # for c in det[:, -1].unique():
                            # n = (det[:, -1] == c).sum()  # detections per class
                            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                # Save image
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


import time
def test():
    start = time.time()
    # get_frames("horizon_1_ship.avi", "output", skip=4)
    d(dataset, device)
    print(time.time() - start)