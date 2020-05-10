import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
from sys import platform

from deep_sort import build_tracker
from utils.draw import draw_boxes, draw_boxes111, draw_id
from yolo_tiny_utils.datasets import *
from yolo_tiny_utils.utils import *

from utils.parser import get_config
from detect import detector

from models import *




class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("请使用gpu测试！", UserWarning)

        # opencv输出界面设置
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        # 是否使用摄像头进行测试
        if args.cam != -1:
            print("使用摄像头 " + str(args.cam))
            self.vdo = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.vdo = cv2.VideoCapture()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        #self.class_names = self.detector.class_names


    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.retrieve()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
            self.vdo.open(self.args.VIDEO_PATH)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)


    def run(self):
        idx_frame = 0
        # Load yolov3_tiny_se detect
        weights = 'best4.pt'
        cfg = 'yolov3-tiny-1cls-se.cfg'
        img_size = 416
        device = torch_utils.select_device(device='0')
        # print(device)

        # Initialize model
        model = Darknet(cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch weights format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:
            load_darknet_weights(model, weights)

        # 预载模型
        model.to(device).eval()
        fps_list = []
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            _, ori_im = self.vdo.read() # get frame
            if ori_im is not None:
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB) # init frame
            #print(im.shape)
            start = time.time()
            # 获取 detection后的结果
            bbox_xywh, cls_conf, cls_ids, bbox_xyxy1 = detector(im, device, model)
            #print(bbox_xywh, cls_conf, cls_ids)


            stop = time.time()

            if bbox_xywh is not None:
                # select car class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # 放大 bbox，避免 bbox过小
                cls_conf = cls_conf[mask]

                # 获取目标 tracking
                outputs, sum_id, track_list = self.deepsort.update(bbox_xywh, cls_conf, im)

                # yolov3 和 DeepSort的可视化
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    # print(bbox_xyxy,identities)
                    ori_im = draw_boxes(ori_im, bbox_xyxy, track_list, identities) # DeepSort
                    ori_im = draw_boxes111(ori_im, bbox_xyxy1, cls_conf) # YOLOv3
                    # ori_im = draw_track(ori_im, bbox_xyxy, track_list) # track
            if bbox_xywh is None:
                sum_car = 'Traffic flow(frame): ' + '0'
                ori_im = cv2.putText(ori_im, sum_car, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, [255, 255, 255], 2)
            ori_im = draw_id(ori_im, sum_id) # sum_id
            end = time.time()
            fps = 1/(end-start+0.001)
            fps_list.append(fps)
            # print("yolov3_tiny-time: {:.03f}s, fps: {:.03f}".format(stop - start, 1 / (stop - start)))
            print("total-time: {:.03f}s, fps: {:.03f}".format(end-start, fps))

            if self.args.display and ori_im is not None:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
        avg_fps = np.mean(fps_list)
        print("avg_fps: {:.03f}".format(avg_fps))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH",default='viedo-03.avi', type=str)
    #parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=1104)
    parser.add_argument("--display_height", type=int, default=622)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1") # 参数不等于 -1 时，开启摄像头
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    #cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
