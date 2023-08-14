# load the opt
import os
import sys
sys.path.append(".")
sys.path.append("..")

import os
from datetime import datetime

from scripts.create_video_dataset import RandomImageGenerator
import matplotlib.pyplot as plt

from habitat.datasets.utils import get_action_shortest_path
import numpy as np
import pickle

from configs.mp3d.options import (
    ArgumentParser,
    get_log_path,
    get_model_path,
    get_timestamp,
)

opts, _ = ArgumentParser().parse()
opts.isTrain = False
split = 'val'
# opts.train_data_path = "/home/jh27kim/code/current_projects/dataset/matterport/dataset/pointnav/mp3d/train/train.json.gz"
opts.val_data_path = "/home/jh27kim/code/current_projects/dataset/matterport/dataset/pointnav/mp3d/val/val.json.gz"
# opts.test_data_path = "/home/jh27kim/code/current_projects/dataset/matterport/dataset/pointnav/mp3d/test/test.json.gz"
opts.scenes_dir = "/home/jh27kim/code/current_projects/dataset/matterport/dataset"

# this is update the path in opts
# from options.options import get_dataset
# Dataset = get_dataset(opts)

image_generator = RandomImageGenerator(
    split,
    opts.render_ids[0],
    opts,
    vectorize=False,
    seed=0,
)

# configs
root_dir = "/home/jh27kim/code/current_projects/dataset/matterport/dataset/val"
samples_before_reset = 100
total_scene = 18 

angle = np.random.uniform(0, 2 * np.pi)
source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
shortest_path_success_distance = 0.2
shortest_path_max_steps = 500

for _i, scene_id in enumerate(range(total_scene)):
    # for each scene, first reset it
    now = datetime.now()
    print(f"{_i} / {total_scene} scene : with id {scene_id}. Time: {now}")

    image_generator.env.reset()
    current_scene = image_generator.env._config.SIMULATOR.SCENE.split("/")[-2]
    # first make the dir for each scene
    os.makedirs(os.path.join(root_dir, current_scene), exist_ok = True)
    
    num_samples = 0
    
    sim = image_generator.env._sim

    while num_samples < samples_before_reset:
        rgb_clip = []
        action_clip = []
        pos_clip = []
        rot_clip = []
        depth_clip = []
        
        source_position = sim.sample_navigable_point()
        target_position = sim.sample_navigable_point()

        shortest_paths = [
                            get_action_shortest_path(
                                sim,
                                source_position=source_position,
                                source_rotation=source_rotation,
                                goal_position=target_position,
                                success_distance=shortest_path_success_distance,
                                max_episode_steps=shortest_path_max_steps,
                            )
                            ]
        # now this sim to get obervation
        shortest_path = shortest_paths[0]

        if len(shortest_path) > 0:
            for index in range(len(shortest_path)):
                
                temp_action = shortest_path[index].action
                temp_pos = shortest_path[index].position
                temp_rot = shortest_path[index].rotation
                temp_img = sim.get_observations_at(temp_pos, temp_rot)['rgb']
                temp_depth = sim.get_observations_at(temp_pos, temp_rot)['depth']
                
                action_clip.append(temp_action)
                rgb_clip.append(temp_img)
                pos_clip.append(temp_pos)
                rot_clip.append(temp_rot)
                depth_clip.append(temp_depth)
        else:
            continue
            
        rgb_clip = np.stack(rgb_clip)
        depth_clip = np.stack(depth_clip)
        action_clip = np.stack(action_clip)
        pos_clip = np.stack(pos_clip)
        rot_clip = np.stack(rot_clip)
        
        file = open(os.path.join(root_dir, current_scene, '%02d.pkl' % num_samples), 'wb')
        video_file = {"rgb":rgb_clip, "depth": depth_clip,"action": action_clip, "pos":pos_clip, "rot": rot_clip}
        pickle.dump(video_file, file)
        file.close()
        
        num_samples += 1 # update the #