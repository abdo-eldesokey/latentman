# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import numpy as np
import torch
import shutil
import pickle
import math
from trimesh import transformations
from tqdm.auto import tqdm


from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.utils.fixseed import fixseed
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.data_loaders.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
from mdm.data_loaders.tensors import collate
from mdm.visualize.vis_utils import npy2obj


from config import get_motion_dir
from misc.io import create_dir

SUPPORTED_ACTIONS = [
    "warm_up",
    "walk",
    "run",
    "jump",
    "drink",
    "lift_dumbbell",
    "sit",
    "eat",
    "turn steering wheel",
    "phone",
    "boxing",
    "throw",
]


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",
    )
    if args.dataset in ["kit", "humanml"]:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


def load_data_and_model(args, num_samples, num_frames):
    ### Load the dataset
    print("Loading dataset...")
    data = load_dataset(args, max_frames=num_samples, n_frames=num_frames)

    ### Load the model
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.cuda()
    model.eval()  # disable random maskin

    return model, diffusion, data


def generate_motion(
    model,
    diffusion,
    data,
    args,
):
    fixseed(args.seed)

    max_frames = 60 if args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if args.dataset == "kit" else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    dist_util.setup_dist(args.device)

    # Create Output directory

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != "":
        texts = [args.text_prompt]
        args.num_samples = 1
        prompt = args.text_prompt
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
        prompt = args.action_name

    out_path = get_motion_dir(prompt, args.seed)

    if not os.path.exists(out_path):
        print(f"Directory {out_path} already exists! Overwriting Files!")
        os.makedirs(out_path)

    assert args.num_samples <= args.batch_size, f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    total_num_samples = args.num_samples * args.num_repetitions

    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}] * args.num_samples
    is_t2m = any([args.input_text, args.text_prompt])
    if is_t2m:
        # t2m
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    else:
        # a2m
        action = data.dataset.action_name_to_action(action_text)
        collate_args = [dict(arg, action=one_action, action_text=one_action_text) for arg, one_action, one_action_text in zip(collate_args, action, action_text)]
    _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f"### Sampling [repetitions #{rep_i}]")

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == "hml_vec":
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == "xyz" else model_kwargs["y"]["mask"].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(
            x=sample,
            mask=rot2xyz_mask,
            pose_rep=rot2xyz_pose_rep,
            glob=True,
            translation=True,
            jointstype="smpl",
            vertstrans=True,
            betas=None,
            beta=0,
            glob_rot=None,
            get_rotations_back=False,
        )

        if args.unconstrained:
            all_text += ["unconstrained"] * args.num_samples
        else:
            text_key = "text" if "text" in model_kwargs["y"] else "action_text"
            all_text += model_kwargs["y"][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "num_samples": args.num_samples,
            "num_repetitions": args.num_repetitions,
        },
    )
    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == "kit" else paramUtil.t2m_kinematic_chain

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            length = all_lengths[rep_i * args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            # save_file = sample_file_template.format(sample_i, rep_i)
            # print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, f"motion_sample_{sample_i}.gif")
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")
    return npy_path, animation_save_path


def smplify(npy_path, rep=-1, gender="neutral"):
    print(f"\nConverting to '{gender}' SMPL Body Models ...")
    npy2obj_ = npy2obj(npy_path, 0, rep, device="0", gender=gender, cuda=True)
    pkl_path = npy_path.replace("npy", "pkl")
    with open(pkl_path, "wb") as handle:
        pickle.dump(npy2obj_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {pkl_path}!")


def export_obj(prompt, seed, task):
    motion_dir = get_motion_dir(prompt, seed)
    pkl_path = f"{motion_dir}/results.pkl"
    with open(pkl_path, "rb") as handle:
        pklobj = pickle.load(handle)

    out_dir = f"{motion_dir}/obj"
    create_dir(out_dir)

    if task == "a2m":
        angle = math.pi
        direction = [1, 0, 0]
        center = [0, 0, 0]
        rot_matrix = transformations.rotation_matrix(angle, direction, center)
    elif task == "t2m":
        rot_matrix = transformations.identity_matrix()
    for idx in tqdm(range(pklobj.num_frames)):
        mesh = pklobj.get_trimesh(0, idx).apply_transform(rot_matrix)
        obj_path = f"{out_dir}/frame_{idx:02d}.obj"
        mesh.export(obj_path)
