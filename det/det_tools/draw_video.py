import cv2
import numpy as np
import os


def make_legend_area(cal_config):
    margin = [50, 50]
    padding = [3,3]
    lane_size = [50, 100]
    margin_mid = 20
    legend_area = []
    solid_lines = []
    dotted_lines = []
    legend_pos = []
    for i in range(int(cal_config['lanes_num'])):
        if i>0 and cal_config['lanes_dir'][i] != cal_config['lanes_dir'][i-1]:
            margin[0] += margin_mid
        p1 =[margin[0]+padding[0]+lane_size[0]*i,margin[1]+padding[1]]
        p2 = [margin[0]+padding[0]+lane_size[0]*i,margin[1]+padding[1]+lane_size[1]]
        p3 = [margin[0]+padding[0]+lane_size[0]*(i+1),margin[1]+padding[1]+lane_size[1]]
        p4 =[margin[0]+padding[0]+lane_size[0]*(i+1),margin[1]+padding[1]]
        legend_area.append([p1,p2,p3,p4])
        # legend_pos.append([int((p3[0] + p1[0]) / 2) - 2* padding[0], int((p3[1] + p1[1]) / 2)+2*padding[1]])
        legend_pos.append([int((p3[0] + p1[0]) / 2) , int((p3[1] + p1[1]) / 2)])
        line1 = [p1,p2]
        line2 = [p4,p3]
        # print(line1)
        # print(line2)
        if line1 in solid_lines:
            dotted_lines.append(line1)
        else:
            solid_lines.append(line1)
        if line2 in solid_lines:
            dotted_lines.append(line2)
        else:
            solid_lines.append(line2)
    margin[0] -= margin_mid
    xyxy = [margin, [margin[0] + int(cal_config['lanes_num'])*lane_size[0] + margin_mid + 2*padding[0], margin[1]+lane_size[1] + 2*padding[1]]]
    return legend_area,xyxy,solid_lines,dotted_lines,legend_pos


def get_legend_blk(cal_config):
    legend_area,xyxy,solid_lines,dotted_lines,legend_pos = make_legend_area(cal_config)
    shape = (cal_config['img_size'][1], cal_config['img_size'][0], 3)
    legend_blk = np.zeros(shape, np.uint8)
    legend_blk[:,:] = np.array([225, 225, 225])

    colors = [(84, 46, 8),
              (0, 30, 255),
              (0, 255, 0),
              (176, 48, 96),
              (255, 215, 0)]
    lanes_num = cal_config['lanes_num']
    for i in range(len(legend_area)):
        area = legend_area[i]
        p1 = np.array(area[0])
        p2 = np.array(area[1])
        p3 = np.array(area[2])
        p4 = np.array(area[3])

        pts = np.array([p1, p2, p3, p4]).reshape(1, 4, 2)
        # print(pts)
        lanes_dir = int(cal_config['lanes_dir'][i])
        if lanes_dir == 1:
            lane_color = colors[lanes_num - i - 1]
        else:
            lane_color = colors[i]
        cv2.fillPoly(legend_blk, pts, color=lane_color)
    for line in solid_lines:
        p0,p1 = line[0],line[1]
        cv2.line(legend_blk, (p0[0], p0[1]), (p1[0], p1[1]), (0, 0, 0), 4)
    # print(dotted_lines)
    for line in dotted_lines:
        p0,p1 = line[0],line[1]
        l = p1[1]-p0[1]
        item_l = int(l/4)
        for i in range(4):
            s = (p0[0],int(p0[1] + item_l*i))
            e = (p1[0],p0[1] + int(item_l*i+item_l/5*2))
            cv2.line(legend_blk, s, e, (255, 255, 255), 4)
    t,b,l,r = xyxy[0][0],xyxy[1][0],xyxy[0][1],xyxy[1][1]
    cv2.line(legend_blk, (t, l), (t, r), (0, 0, 0), 6)
    cv2.line(legend_blk, (t, r), (b, r), (0, 0, 0), 6)
    cv2.line(legend_blk, (b, r), (b, l), (0, 0, 0), 6)
    cv2.line(legend_blk, (b, l), (t, l), (0, 0, 0), 6)
    return legend_blk,xyxy,legend_pos


def get_cal_blk(cal_config, area_points_i, cal_points_i_dw,cal_points_i_dh, lanes_area_i,speed_measure_lines):
    shape = (cal_config['img_size'][1], cal_config['img_size'][0], 3)
    cal_blk = np.zeros(shape, np.uint8)
    colors = [(84, 46, 8),
              (0, 30, 255),
              (0, 255, 0),
              (176, 48, 96),
              (255, 215, 0)]
    # for point in area_points_i:
    #     cv2.circle(cal_blk, (point[0], point[1]), 10, (0, 0, 255), -1)
    lanes_num = cal_config['lanes_num']
    for i in range(int(len(lanes_area_i) / 4)):
        point1 = np.array(lanes_area_i[i * 4])
        point2 = np.array(lanes_area_i[i * 4 + 1])
        point3 = np.array(lanes_area_i[i * 4 + 2])
        point4 = np.array(lanes_area_i[i * 4 + 3])
        pts = np.array([point1, point2, point3, point4]).reshape(1, 4, 2)
        # print(pts)
        lanes_dir = int(cal_config['lanes_dir'][i])
        if lanes_dir == 1:
            lane_color = colors[lanes_num - i - 1]
        else:
            lane_color = colors[i]
        cv2.fillPoly(cal_blk, pts, color=lane_color)
    for line in speed_measure_lines:
        point1 = line[0]
        point2 = line[1]
        cv2.line(cal_blk, point1, point2, (0, 0, 0), 1)
    # for i in range(int(len(cal_points_i_dw) / 2)):
    #     point1 = cal_points_i_dw[i * 2]
    #     point2 = cal_points_i_dw[i * 2 + 1]
    #     cv2.line(cal_blk, (point1[0], point1[1]), (point2[0], point2[1]), (0, 255, 255), 2)
    # im0 = cv2.addWeighted(im0, 1.0, blk, 0.4, 1)
    # for i in range(int(len(cal_points_i_dh) / 2)):
    #     point1 = cal_points_i_dh[i * 2]
    #     point2 = cal_points_i_dh[i * 2 + 1]
    #     cv2.line(cal_blk, (point1[0], point1[1]), (point2[0], point2[1]), (0, 255, 0), 2)
    return cal_blk


def draw_det(annotator1,save_dir, frame):
    im0 = annotator1.result()
    img_dir = os.path.join(save_dir, 'det_img')
    os.makedirs(img_dir, exist_ok=True)
    cv2.imwrite(os.path.join(img_dir, f'frame_{frame}.jpg'), im0)


def draw_video(annotator,traj_points, view_img, p, save_img,
               vid_path, save_path, save_dir, frame,vid_writer,
               vid_cap, show_traj,legend_pos,lane_count):
    im0 = annotator.result()
    img_dir = os.path.join(save_dir,'img')
    os.makedirs(img_dir,exist_ok=True)
    if show_traj:
        for i in range(len(traj_points)):
            im0 = cv2.circle(im0, (int(traj_points[i][0]),int(traj_points[i][1])), 2, (0, 0, 255), 2)

    for i in range(len(legend_pos)):
        pos = legend_pos[i]
        text_lane_count = str(lane_count[i])
        (base_width, base_h), bottom = cv2.getTextSize(text_lane_count, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(im0,text_lane_count, (pos[0]-int(base_width/2), pos[1]+int(base_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if view_img:
        cv2.imshow(str(p), im0)
        cv2.moveWindow(str(p), 100, 100)
        cv2.waitKey(1)  # 1 millisecond
    if save_img:
        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                # fps = 30
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path += '.mp4'
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(im0)
        cv2.imwrite(os.path.join(img_dir,f'frame_{frame}.jpg'), im0)
    return vid_path, vid_writer