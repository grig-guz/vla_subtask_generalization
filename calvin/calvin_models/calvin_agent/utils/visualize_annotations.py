import logging

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from sklearn.manifold import TSNE
from tqdm import tqdm

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)

select_idx = 0

def determine_low_level_task_seq(env, datapoint, start_timestep, seq_length, low_level_tasks):
    # Need to check that only a particular sequence of low-level tasks gets completed.
    # Iterate over all   
    #print(datapoint['state_info']['scene_obs'][start_timestep])
    env.reset(robot_obs=datapoint['state_info']['robot_obs'][start_timestep], 
              scene_obs=datapoint['state_info']['scene_obs'][start_timestep])
    start_info = env.get_info()
    end_info = {} 
    subtasks_completed = []
    
    for tstep in range(start_timestep + 1, seq_length):
        env.reset(robot_obs=datapoint['state_info']['robot_obs'][tstep], 
                scene_obs=datapoint['state_info']['scene_obs'][tstep])
        #print(datapoint['state_info']['scene_obs'][tstep])
        end_info = env.get_info()

        tasks_completed = low_level_tasks.get_task_info(start_info, end_info)
        #print()
        #print(f"Timestep: {tstep}, tasks found: {tasks_completed}")
        #print()
        if len(tasks_completed) == 0:
            continue
        elif len(tasks_completed) > 1:
            return False, []
        else:
            subtasks_completed.append(list(tasks_completed)[0])
            start_info = end_info

    return True, subtasks_completed
    


def generate_single_seq_gif(seq_img, seq_length, imgs, idx, i, data, env, datapoint, tasks, low_level_tasks):
    s, c, h, w = seq_img.shape
    seq_img = np.transpose(seq_img, (0, 2, 3, 1))
    print("Seq length: {}".format(s))
    print("From: {} To: {}".format(idx[0], idx[1]))
    print("Instruction: ", data['language']['task'][i], data['language']['ann'][i])
    """
    if i == 18: #'ungrasp' in data['language']['task'][i]:# and 'pink' in data['language']['task'][i] and i > 90:
        all_info_dicts = []
        for tstep in range(seq_length):
            print("The timestep: ", tstep)
            env.reset(robot_obs=datapoint['state_info']['robot_obs'][tstep],
                    scene_obs=datapoint['state_info']['scene_obs'][tstep])
            info_dict = env.get_info()
            all_info_dicts.append(info_dict)
            #print(info_dict['scene_info'])
            print(datapoint['state_info']['scene_obs'][tstep])
            obj_contacts = {}
            for obj in info_dict['scene_info']['movable_objects'].keys():
                obj_contacts[obj] = set(c[2] for c in info_dict['scene_info']['movable_objects'][obj]['contacts'])
            #print(obj_contacts)
        #print("Full dict: ", info_dict)
        #print(tasks.get_task_info(all_info_dicts[0], all_info_dicts[-1]))
        _, subtasks_completed = determine_low_level_task_seq(env, datapoint, start_timestep=0, seq_length=seq_length, low_level_tasks=low_level_tasks)
        #print("The subtasks completed: ", subtasks_completed)
        breakpoint()
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    for j in range(seq_length):
        imgRGB = seq_img[j]
        imgRGB = cv2.resize(
            ((imgRGB - imgRGB.min()) / (imgRGB.max() - imgRGB.min()) * 255).astype(np.uint8), (500, 500)
        )
        # img = plt.imshow(imgRGB, animated=True)
        # text1 = plt.text(
        #     200, 200, f"t = {j}", ha="center", va="center", size=10, bbox=dict(boxstyle="round", ec="b", lw=2)
        # )
        img = cv2.putText(imgRGB, f"t = {j}", (350, 450), font, color=(0, 0, 0), fontScale=1, thickness=2)
        img = cv2.putText(
            img, f"{i}. {data['language']['subtasks_actual'][i]}", (10, 30), font, color=(0, 0, 0), fontScale=0.25, thickness=1
        )
        img = cv2.putText(
            img, f"{i}. {data['language']['ann'][i]}", (100, 20), font, color=(0, 0, 0), fontScale=0.5, thickness=1
        )[:, :, ::-1]


        # text = plt.text(
        #     100,
        #     20,
        #     f"{i}. {data['language']['ann'][i]}",
        #     ha="center",
        #     va="center",
        #     size=10,
        #     bbox=dict(boxstyle="round", ec="b", lw=2),
        # )
        
        if j == 0:
            for _ in range(25):
                imgs.append(img)
        imgs.append(img)
    return imgs


def generate_all_seq_gifs(data, dataset, env, tasks, low_level_tasks):
    imgs = []
    # fig = plt.figure()
    print("All indices:", len(data["info"]["indx"]))
    for i, idx in enumerate(tqdm(data["info"]["indx"][select_idx:])):
        #if 'place' in data["language"]["task"][i]:
        seq_length = idx[1] - idx[0]
        dataset.max_window_size, dataset.min_window_size = seq_length, seq_length

        #start = dataset.episode_lookup.tolist().index(idx[0])
        #print("THE START TIMESTEP and index: ", start, idx)
        seq_img = dataset[start]["rgb_obs"]["rgb_static"].numpy()
        # if 'lift' in data['language']['task'][i]:
        imgs = generate_single_seq_gif(seq_img, seq_length, imgs, idx, i, data, env, dataset[start], tasks, low_level_tasks)
    return imgs


def load_data(cfg):
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.train_dataloader()["vis"].dataset
    tasks = hydra.utils.instantiate(cfg.callbacks.rollout.tasks)
    low_level_tasks = hydra.utils.instantiate(cfg.callbacks.rollout.low_level_tasks)

    file_name = dataset.abs_datasets_dir / cfg.lang_folder / "auto_lang_ann.npy"
    return np.load(file_name, allow_pickle=True).reshape(-1)[0], dataset, tasks, low_level_tasks


def plot_and_save_gifs(imgs):
    # anim = ArtistAnimation(fig, imgs, interval=75)
    # plt.axis("off")
    # plt.title("Annotated Sequences")
    # plt.show()
    # anim.save("/tmp/summary_lang_anns.mp4", writer="ffmpeg", fps=15)
    video = cv2.VideoWriter("/ubc/cs/research/nlp/grigorii/projects/low_level_tasks/summary_lang_anns.avi", cv2.VideoWriter_fourcc(*"XVID"), 15, (500, 500))
    for img in imgs:
        video.write(img)
    video.release()


def generate_task_id(tasks):
    labels = list(sorted(set(tasks)))
    task_ids = [labels.index(task) for task in tasks]
    return task_ids


def visualize_embeddings(data, with_text=True):
    emb = data["language"]["emb"].squeeze()
    tsne_emb = TSNE(n_components=2, random_state=40, perplexity=20.0).fit_transform(emb)

    emb_2d = tsne_emb

    task_ids = generate_task_id(data["language"]["task"])

    cmap = ["orange", "blue", "green", "pink", "brown", "black", "purple", "yellow", "cyan", "red", "grey", "olive"]
    ids_in_legend = []
    for i, task_id in enumerate(task_ids):
        if task_id not in ids_in_legend:
            ids_in_legend.append(task_id)
            plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=cmap[task_id], label=data["language"]["task"][i])
            if with_text:
                plt.text(emb_2d[i, 0], emb_2d[i, 1], data["language"]["ann"][i])
        else:
            plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=cmap[task_id])
            if with_text:
                plt.text(emb_2d[i, 0], emb_2d[i, 1], data["language"]["ann"][i])
    plt.legend()
    plt.title("Language Embeddings")
    plt.show()


@hydra.main(config_path="../../conf", config_name="lang_ann_visualization.yaml")
def main(cfg: DictConfig) -> None:
    data, dataset_obj, tasks, low_level_tasks = load_data(cfg)
    scene_idx_info = np.load(dataset_obj.abs_datasets_dir / "scene_info.npy", allow_pickle=True).item()
    # visualize_embeddings(data)
    envs = {
        scene: hydra.utils.instantiate(
            cfg.callbacks.rollout_lh.env_cfg, dataset_obj, "cuda:0", scene=scene
        )
        for scene, _ in scene_idx_info.items()
    }
    imgs = generate_all_seq_gifs(data, dataset_obj, envs['calvin_scene_D'], tasks, low_level_tasks)
    plot_and_save_gifs(imgs)


if __name__ == "__main__":
    main()
