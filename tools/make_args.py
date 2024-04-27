import argparse


def make_args():
    parser = argparse.ArgumentParser(description="highwayTrack")
    parser.add_argument("-n", "--name", type=str, default='demo.mp4')
    parser.add_argument("-nv",
                        "--non_visual",
                        action="store_true")
    parser.add_argument("-udl",
                        "--use_det_labels",
                        action="store_true")
    parser.add_argument("-e",
                        "--eval",
                        action="store_true")
    parser.add_argument("-to",
                        "--track_only",
                        action="store_true")
    parser.add_argument("-cco",
                        "--camera_cal_only",
                        action="store_true")
    return parser.parse_args()
