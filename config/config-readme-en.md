## video_type
  Video type

## img_size
  Video frame size

## reverse
  Whether the image is reversed. By default, the camera is on the left side of the lane. Set to false if it is on the right.

## border_gap
  The minimum gap between the vehicle bounding box and the frame edge, adjust based on the video frame size.

## det_config
  - **cls_min_box_area**

    Minimum bounding box size for detected cars, vans, and trucks.

  - **max_frame**
    
    The number of frames to process in the camera calibration section. Decrease this number in cases of high traffic flow.
    
  - **Other settings are default parameters of the `yolov5` model, and it is not recommended to modify them.**

## track_config:
  - **frame_rate**
    
    Video frame rate
    
  - **Other settings are default parameters of the `byteTrack` model, and it is not recommended to modify them.**

## area_division
  - **model_type**
    
    Uses the `sam_vit_h_4b8939.pth` pretrained model parameters by default, modifications not recommended.
    
## camera_cal_config:
  - **max_dotted_line_num**
    
    Maximum number of recognized dashed lines, default is 4, not recommended to modify.
    
  - **dotted_line_len**
    
    Actual length of the lane dashed line, in meters.
    
  - **dotted_line_gap_len**
    
    Gap length between lane dashed lines, in meters.
    
  - **lane_width**
    
    Lane width, in meters.

## lane_line_completion:
  Actual relative positions of the lane lines, positions of each lane line from left to right relative to the first recognizable dashed line on the far left (if the camera is on the right side of the lane, it refers to after the image is reversed), in meters.
