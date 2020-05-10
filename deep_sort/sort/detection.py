import numpy as np
# 单个bbox的检测单元
class Detection(object):
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    # 将bbox格式转换为（左上角，右下角），即(x, y, x+w, y+h)
    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    # 将bbox格式转换为（中心横坐标，中心纵坐标，宽高比）
    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
