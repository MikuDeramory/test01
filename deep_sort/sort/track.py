# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    # 不确定态，这种状态会在初始化一个Track的时候分配，并且只有在连续匹配上n_init帧才会转变为确定态。
    # 如果在处于不确定态的情况下没有匹配上任何detection，那将转变为删除态。
    Confirmed = 2
    #  确定态，代表该Track确实处于匹配状态。
    #  如果当前Track属于确定态，但是失配连续达到max age次数的时候，就会被转变为删除态。
    Deleted = 3
    # 删除态，说明该Track已经失效。

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        # max age是一个存活期限，默认为70帧
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        # hits和 n_init进行比较
        # hits在每次进行 updata时进行一次更新（match时进行updata）
        # hits代表匹配上了多少次，若匹配次数超过n_init就会设置成confirmed转态
        self.age = 1
        self.time_since_update = 0
        # time_sine_updata 每次调用 predict函数时 +1
        # 每次调用 updata函数时设置成 0

        self.state = TrackState.Tentative
        self.features = []
        # 每一个 track对应多个 features，每次更新都将最新的 feature添加到列表中
        # 保存列表的目的是解决目标被遮挡后的问题，需要从以往帧对应的特征进行匹配。
        # 如果特征过多会降低计算速度，因此用参数 budget控制 features的长度。
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        # 如果连续 n_init帧都没出现适配，就将其设置为 deleted状态
        self._max_age = max_age
        # 失配次数上限

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
