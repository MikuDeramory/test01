import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# 根据不同类分配颜不同颜色
def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, track_list, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        #print(identities[i])

        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1+5,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, (255,180,20), 2)

        # box center track
        center_track = track_list[identities[i]]
        for i in range(1, len(center_track)):
            thickness = int(np.sqrt(64 / float(i + 1)) * 2)
            cv2.line(img,center_track[i - 1], center_track[i], color, thickness)

    return img

def draw_boxes111(img, bbox, identities=None, offset=(0,0)):
    now_car = 0
    for i,box in enumerate(bbox):
        # print(box)
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        now_car+=1
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img, 'car', (x1+30, y1+t_size[1]-25), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    sum_car = 'Traffic flow(frame): '+ str(now_car)
    cv2.putText(img, sum_car, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, [255,255,255], 2)
    return img

def draw_id(img, id):
    sum_id = 'Traffic flow(total):'+ str(id)
    cv2.putText(img, sum_id, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2.5, [255,255,255], 2)
    return img

def draw_frame_null(img):
    sum_car = 'Traffic flow(frame): ' + '0'
    cv2.putText(img, sum_car, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, [255, 255, 255], 2)
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
