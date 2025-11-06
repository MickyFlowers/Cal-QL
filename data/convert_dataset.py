import argparse
import glob
import io
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
from tqdm import tqdm


def _process_one_folder(folder: str, output_dir: str):
    """Process a single folder: read poses.npy, video.mp4 and write hdf5.

    Returns:
        (joint_max, joint_min) as np.ndarray (shape [D])
    """
    folder_path = Path(folder)
    h5_file = Path(output_dir) / (folder_path.name + ".hdf5")
    video_file = folder_path / 'video.mp4'
    robot_data_file = folder_path / 'poses.npy'

    robot_data = np.load(robot_data_file, allow_pickle=True).item()
    joint_positions = robot_data['joint_positions']  # (T, D)

    # Video reading
    clip = VideoFileClip(str(video_file))
    n_frames = clip.reader.nframes  # avoid materializing list of frames twice

    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('joint_positions', data=joint_positions)
        png_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
        hf.create_dataset('video', (n_frames,), dtype=png_dtype)
        for i, frame in enumerate(clip.iter_frames()):
            img = Image.fromarray(frame)
            with io.BytesIO() as output:
                img.save(output, format='PNG')
                png_data = output.getvalue()
            hf['video'][i] = np.frombuffer(png_data, dtype='uint8')
    clip.close()
    return joint_positions.max(axis=0), joint_positions.min(axis=0)


def main(args):
    folders = glob.glob(os.path.join(args.input_path, '*/'))
    print(len(folders), "folders found in", args.input_path)
    output_path = Path(args.input_path).parent / (Path(args.input_path).name + '-converted')
    output_path.mkdir(exist_ok=True)

    # Multiprocessing over folders
    num_workers = args.num_workers if args.num_workers > 0 else multiprocessing.cpu_count()
    joint_max_global = None
    joint_min_global = None

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one_folder, f, str(output_path)): f for f in folders}
        for fut in tqdm(as_completed(futures), total=len(folders), desc='Processing folders'):
            jmax, jmin = fut.result()
            if joint_max_global is None:
                joint_max_global = jmax
                joint_min_global = jmin
            else:
                joint_max_global = np.maximum(joint_max_global, jmax)
                joint_min_global = np.minimum(joint_min_global, jmin)

    json.dump({
        'joint_max': joint_max_global.tolist(),
        'joint_min': joint_min_global.tolist(),
    }, open(output_path / 'statistics.json', 'w'), indent=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert dataset format.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input dataset root directory.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of parallel processes (0 = use cpu_count).')
    args = parser.parse_args()
    main(args)