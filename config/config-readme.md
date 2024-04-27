## video_type
  视频类型
## img_size
  视频画面大小
## reverse
  画面是否翻转，默认摄像头位于车道左侧，如位于右侧则设置为false
## border_gap
  车辆包围盒与画面边框的最小间距，根据视频画面大小调整
## det_config
  - **cls_min_box_area**

    轿车、厢式货车、大货车被检测器检测的最小包围盒大小
    
  - **max_frame**
    
    相机标定部分需要处理的帧数，车流量大的情况调小
    
  - **其他为`yolov5` 模型默认参数，不建议修改**
## track_config:
  - **frame_rate**
    
    视频帧率
    
  - **其他为`byteTrack`默认参数，不建议修改**
## area_division
  - **model_type**
    
    默认使用sam_vit_h_4b8939.pth预训练模型参数，不建议修改
    
## camera_cal_config:
  - **max_dotted_line_num**
    
    最大虚线识别数，默认4，不建议修改
    
  - **dotted_line_len**
    
    车道虚线真实长度，单位m
    
  - **dotted_line_gap_len**
    
    车道虚线间隔长度，单位m
    
  - **lane_width**
    
    车道宽度，单位m
    
## lane_line_completion:
  车道线真实相对位置，从左往右各车道线位置，相对于最左侧第一条可识别虚线（如摄像头位于车道右侧，则为画面翻转后最左侧第一条），单位m
