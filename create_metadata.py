import json
import glob
import os
import tqdm
import pickle


def main(args):
    clip_paths = {
        "paths": [],
    }
    
    clip_length = args.length + (args.gap - 1) * (args.length - 1)
    scene_paths = glob.glob(os.path.join(args.data_path, "*"))

    for i, scene_path in enumerate(scene_paths):
        clip_list = sorted(glob.glob(os.path.join(scene_path, "*")))
        print(i, " / ", len(scene_paths))
        for clip_path in clip_list:
            print(clip_path)
            file = open(clip_path, 'rb')
            video_clip = pickle.load(file)
            file.close()
            
            if video_clip['rgb'].shape[0] >= clip_length:
                # filter the data which is too short
                clip_paths["paths"].append(clip_path)

    clip_paths["total_clips"] = len(clip_paths["paths"])
    
    save_path = os.path.join(args.data_path, f"length_{args.length}_gap_{args.gap}_valid_clip_path.json")
    with open(save_path, 'w') as f:
        json.dump(clip_paths, f, indent=2)
    
    print("Saved json at ", save_path)

    with open(save_path, "r") as json_file:
        verification_paths = json.load(json_file)
    
    assert verification_paths["total_clips"] == clip_paths["total_clips"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--gap", type=int, required=True)
    # Default length=2 gap=3

    args = parser.parse_args()

    main(args)


