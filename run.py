from s0_vehicle_detection import vehicle_detection
from s1_vehicle_pre_track import vehicle_pre_track
from s2_pre_process import pre_process
from s3_lane_area_division import lane_area_division
from s4_lane_line_extraction import lane_line_extraction
from s5_camera_cal import camera_cal
from s6_lane_line_completion import lane_line_completion
from s7_highwayTrack import vehicle_highway_track
from s8_eval import track_eval
from tools.tools_init import *


def pre_det_track_and_process(file_name, visualization=False):
    vehicle_detection(file_name)
    vehicle_pre_track(file_name)
    pre_process(file_name,visualization=visualization)


def camera_calibration(file_name, visualization=False):
    camera_cal(file_name,visualization=visualization)
    lane_line_completion(file_name,visualization=visualization)


def track_and_eval(file_name,args):
    vehicle_highway_track(file_name,use_det_labels=args.use_det_labels)
    if args.eval:
        track_eval(file_name)


def run_all(args):
    file_name = args.name
    visualization = not args.non_visual
    if not args.track_only:
        pre_det_track_and_process(file_name, visualization=visualization)
        lane_area_division(file_name,visualization=visualization)
        lane_line_extraction(file_name, visualization=visualization)
        camera_calibration(file_name, visualization=visualization)
    if not args.camera_cal_only:
        track_and_eval(file_name,args)


if __name__ == '__main__':
    run_all(make_args())

