import os

import utils


def main():
    base_folder = "dataset"
    load_folder = "external"
    output_folder = "processed"

    load_path = os.path.join(".", base_folder, load_folder)
    output_path = os.path.join(".", base_folder, output_folder)

    test_dataset = utils.datasets.TestTimeAugCIFAR10(load_path)
    test_dataset.aug(root=output_path, transform_name="flip_crop", aug_num=10)


if __name__ == "__main__":
    main()
