from TrackEval.eval import *
from tools.tools_init import *


def process_data(input_file, output_file, mode=1):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        if line[0] == '#':
            continue
        parts = line.strip().split()
        if mode == 1:
            f, id, x1, y1, x2, y2 = parts
            # f = int(f)
        elif mode == 2:
            id, f, x1, y1, x2, y2, _, _, _, _ = parts
            # f = int(f) - 1
        else:
            id, f, x1, y1, x2, y2, _, _ = parts
            # f = int(f)
        f = int(f)
        x1, y1 = float(x1), float(y1)
        x2, y2 = float(x2), float(y2)
        w = float(x2) - float(x1)
        h = float(y2) - float(y1)
        new_line = f"{f},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
        processed_lines.append(new_line)

    with open(output_file, 'w') as f:
        f.writelines(processed_lines)


def prepare_seq_info(seq_info, configs, seq_maps):
    os.makedirs(os.path.dirname(seq_maps), exist_ok=True)
    par_path = os.path.dirname(seq_info)
    name = par_path.split('\\')[-1]
    # Constants
    im_dir = "img1"
    frame_rate = configs['track_config']['frame_rate']
    im_width = configs['img_size'][0]
    im_height = configs['img_size'][1]
    im_ext = ".jpg"

    # Read the input file to find the maximum frame id for seqLength
    max_id = configs['total_frames']
    # Prepare the seqinfo.ini content
    seqinfo_content = f"""[Sequence]
name={name}
imDir={im_dir}
frameRate={frame_rate}
seqLength={max_id}
imWidth={im_width}
imHeight={im_height}
imExt={im_ext}
"""
    # Save the seqinfo.ini in the output folder
    with open(seq_info, 'w') as file:
        file.write(seqinfo_content)
    with open(seq_maps, 'w') as file:
        file.write('name\n')
        file.write(f'{name}\n')


def init_data(file_name,logger):
    path_dict = get_path_dict(file_name)
    configs = load_config(file_name)
    gt_ori = path_dict['gt-ori']
    gt_final = path_dict['gt-final']
    seq_info = path_dict['seq-info']
    seq_maps = path_dict['seq-maps']

    process_data(gt_ori, gt_final)
    prepare_seq_info(seq_info, configs, seq_maps)
    process_data(path_dict['highwayTrack-online'], path_dict['highwayTrack-online-mot17'], mode=2)
    # process_data(path_dict['highwayTrack-base-interpolation'], path_dict['highwayTrack-base-interpolation-mot17'], mode=1)
    # process_data(path_dict['byteTrack-realtime'], path_dict['byteTrack-realtime-mot17'], mode=3)
    # process_data(path_dict['byteTrack-base-interpolation'], path_dict['byteTrack-base-interpolation-mot17'], mode=1)


def eval_by_tracker_name(tracker_name, file_name):
    file_name = file_name.split('.')[0]
    path_dict = get_path_dict(file_name)
    args_eval = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    }
    args_dataset = {
        'GT_FOLDER': path_dict['gt-folder'],
        'TRACKERS_FOLDER': path_dict[f'{tracker_name}-folder'],
        'OUTPUT_FOLDER': path_dict[f'{tracker_name}-res'],
        'BENCHMARK': 'exp',
        'SPLIT_TO_EVAL': 'test',
        'TRACKERS_TO_EVAL': [file_name],
        'DO_PREPROC': False,
        'PRINT_CONFIG': False,
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt.txt',
    }
    args_metrics = {
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'THRESHOLD': 0.5,
        'PRINT_CONFIG': False,
    }
    output_res, _ = eval(args_eval, args_dataset, args_metrics)
    data = output_res['MotChallenge2DBox'][file_name][file_name]['vehicle']
    mot_metrics = {
        'Tracker': tracker_name,
        'HOTA': np.mean(data['HOTA']['HOTA']),
        'MOTA': data['CLEAR']['MOTA'],
        'IDF1': data['Identity']['IDF1'],
        'MOTP': data['CLEAR']['MOTP'],
        'TP': int(data['CLEAR']['CLR_TP']),
        'FN': int(data['CLEAR']['CLR_FN']),
        'FP': int(data['CLEAR']['CLR_FP']),
        'IDSW': int(data['CLEAR']['IDSW']),
        'MT': int(data['CLEAR']['MT']),
        'PT': int(data['CLEAR']['PT']),
        'ML': int(data['CLEAR']['ML']),
        # 'IDTP': int(data['Identity']['IDTP']),
        # 'IDFN': int(data['Identity']['IDFN']),
        # 'IDFP': int(data['Identity']['IDFP']),
        'IDs': int(data['Count']['IDs']),
        'GT_IDs': int(data['Count']['GT_IDs']),
    }

    def format_values(value):
        if isinstance(value, float):
            return round(value * 100, 2)
        return value

    return {key: format_values(value) for key, value in mot_metrics.items()}


def print_table_from_dicts(dict_list):
    # 获取所有的键作为列名
    column_names = list(dict_list[0].keys())
    first_width = 0
    for dict_ in dict_list:
        first_width = max(first_width,len(dict_['Tracker']))
    # 针对列宽的设置：第一列15个字符，其他列6个字符
    column_widths = [first_width] + [5] * (len(column_names) - 1)

    # 格式化列名，根据各列的宽度设置
    header_parts = [f"{name:<{column_widths[i]}}" for i, name in enumerate(column_names)]
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))
    for dict_ in dict_list:
        # 格式化并打印数据行
        row_parts = [f"{dict_[name]:<{column_widths[i]}.2f}" if isinstance(dict_[name], float)
                     else f"{dict_[name]:<{column_widths[i]}}" for i, name in enumerate(column_names)]
        row = " | ".join(row_parts)
        print(row)


def track_eval(file_name):
    logger = get_logger('eval')
    path_dict = get_path_dict(file_name)
    gt_ori = path_dict['gt-ori']
    if not os.path.exists(gt_ori):
        logger.info('No gt file for eval...')
        return
    init_data(file_name,logger)
    highwayTrack_online_metrics = eval_by_tracker_name('highwayTrack-online', file_name)
    print_table_from_dicts(
        [highwayTrack_online_metrics])


if __name__ == '__main__':
    args = make_args()
    track_eval(args.name)
