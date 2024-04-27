class TrackerArgs(object):
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, aspect_ratio_thresh=1.6, min_box_area=400,
                 img_height=1080, img_width=1920, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.img_height = img_height
        self.img_width = img_width
        self.mot20 = mot20