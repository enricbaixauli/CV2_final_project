from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_test_dataset(source_root: Path, destination_root: Path, image_folder_name: str = "image_01") -> int:
    """Copy every file inside each `image_01` folder into one flat destination folder.

    Folders are grouped by person, so outputs are named like `p01_image_01.png`
    and `p01_image_01.mat`, `p01_image_02.png`, and so on.
    """

    copied_files = 0
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()

    destination_root.mkdir(parents=True, exist_ok=True)
    for item in destination_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for person_folder in sorted(source_root.glob("user_*/")):
        if not person_folder.is_dir():
            continue

        user_suffix = person_folder.name.split("_")[-1]
        person_name = f"p{int(user_suffix):02d}"
        image_index = 1

        for image_folder in sorted(person_folder.rglob(image_folder_name)):
            if not image_folder.is_dir():
                continue

            base_name = f"{person_name}_image_{image_index:02d}"
            image_index += 1

            for item in image_folder.iterdir():
                if item.is_file():
                    target_name = f"{base_name}{item.suffix.lower()}"
                    shutil.copy2(item, destination_root / target_name)
                    copied_files += 1

    return copied_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a test dataset by copying the contents of each image_01 folder."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("dataset"),
        help="Root folder that contains the user_01 to user_12 subfolders.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path("dataset/test"),
        help="Folder where the copied dataset will be created.",
    )
    parser.add_argument(
        "--image-folder-name",
        type=str,
        default="image_01",
        help="Name of the folder to copy inside each position_index_* directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = build_test_dataset(
        source_root=args.source_root,
        destination_root=args.destination_root,
        image_folder_name=args.image_folder_name,
    )
    print(f"Copied {total} files into {args.destination_root.resolve()}")


if __name__ == "__main__":
    main()