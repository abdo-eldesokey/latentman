import os
from pathlib import Path

CWD = os.getcwd()
MDM_PATH = f"{CWD}/external/mdm"
DENSEPOSE_PATH = f"{CWD}/external/detectron2/projects/DensePose"

SMPL_DATA_PATH = f"{MDM_PATH}/body_models/smpl"
A2M_MODEL_PATH = f"{MDM_PATH}/save/humanact12/model000350000.pt"
T2M_MODEL_PATH = f"{MDM_PATH}/save/humanml_trans_enc_512/model000200000.pt"

DATASETS_PATH = f"{MDM_PATH}/dataset"

WORKSPACE_PATH = f"{CWD}/workspace"
if not Path(WORKSPACE_PATH).exists():
    Path(WORKSPACE_PATH).mkdir()

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, "J_regressor_extra.npy")

ROT_CONVENTION_TO_ROT_NUMBER = {
    "legacy": 23,
    "no_hands": 21,
    "full_hands": 51,
    "mitten_hands": 33,
}

GENDERS = ["neutral", "male", "female"]
NUM_BETAS = 10


def get_motion_dir(action: str, seed=None):
    if action[-1].isdigit():  # A path to directory is passed
        return f"{WORKSPACE_PATH}/{action}"
    else:
        return f"{WORKSPACE_PATH}/{action.replace(' ', '_')}_{seed}"
