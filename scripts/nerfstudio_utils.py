import os
import shutil
import gdown
import zipfile
from pathlib import Path

SAVING_DEST = '/scratch/gpfs/yb6751/datasets'


def grab_file_id(zip_url: str) -> str:
    """Get the file id from the google drive zip url."""
    s = zip_url.split("/d/")[1]
    return s.split("/")[0]

nerfstudio_file_ids = {
    "bww_entrance": grab_file_id("https://drive.google.com/file/d/1ylkRHtfB3n3IRLf2wplpfxzPTq7nES9I/view?usp=sharing"),
    "campanile": grab_file_id("https://drive.google.com/file/d/13aOfGJRRH05pOOk9ikYGTwqFc2L1xskU/view?usp=sharing"),
    "desolation": grab_file_id("https://drive.google.com/file/d/14IzOOQm9KBJ3kPbunQbUTHPnXnmZus-f/view?usp=sharing"),
    "dozer": grab_file_id("https://drive.google.com/file/d/1-OR5F_V5S4s-yzxohbwTylaXjzYLu8ZR/view?usp=sharing"),
    "library": grab_file_id("https://drive.google.com/file/d/1Hjbh_-BuaWETQExn2x2qGD74UwrFugHx/view?usp=sharing"),
    "poster": grab_file_id("https://drive.google.com/file/d/1dmjWGXlJnUxwosN6MVooCDQe970PkD-1/view?usp=sharing"),
    "redwoods2": grab_file_id("https://drive.google.com/file/d/1rg-4NoXT8p6vkmbWxMOY6PSG4j3rfcJ8/view?usp=sharing"),
    "storefront": grab_file_id("https://drive.google.com/file/d/16b792AguPZWDA_YC4igKCwXJqW0Tb21o/view?usp=sharing"),
    "vegetation": grab_file_id("https://drive.google.com/file/d/1wBhLQ2odycrtU39y2akVurXEAt9SsVI3/view?usp=sharing"),
    "all": None,
}

def download_capture_name(save_dir: Path, dataset_name: str, capture_name: str, capture_name_to_file_id: dict):
    """Download specific captures a given dataset and capture name."""

    url = f"https://drive.google.com/uc?id={capture_name_to_file_id[capture_name]}"
    target_path = str(save_dir / f"{dataset_name}/{capture_name}")
    os.makedirs(target_path, exist_ok=True)
    download_path = Path(f"{target_path}.zip")
    tmp_path = str(save_dir / ".temp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    gdown.download(url, output=str(download_path))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    inner_folders = os.listdir(tmp_path)
    assert len(inner_folders) == 1, "There is more than one folder inside this zip file."
    folder = os.path.join(tmp_path, inner_folders[0])
    shutil.rmtree(target_path)
    shutil.move(folder, target_path)
    shutil.rmtree(tmp_path)
    os.remove(download_path)


if __name__ == "__main__":
    for capture_name in nerfstudio_file_ids:
        if capture_name != "all":
            download_capture_name(Path(SAVING_DEST), "nerfstudio", capture_name, nerfstudio_file_ids)