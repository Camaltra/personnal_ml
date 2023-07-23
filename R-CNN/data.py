from abc import abstractmethod, ABC
import os

import numpy as np
import pandas as pd
import cv2
from image_tools_process import GroundTruthBB, process_iou, SelectiveSBB
from constants import class_name_to_class_num
import tensorflow as tf
import shutil


class FileManager:
    def __init__(self, data_path):
        self.data_path = data_path
        self.base_directory = os.getcwd()

    @staticmethod
    def _create_folder(folder_path):
        os.mkdir(folder_path)

    @staticmethod
    def _delete_folder(folder_path):
        shutil.rmtree(folder_path, ignore_errors=False, onerror=None)

    def reset_folder(self, folder_path):
        try:
            self._delete_folder(folder_path)
        except Exception:
            print("Error while deleting folder <%s>, skipping the creation", folder_path)
        finally:
            self._create_folder(folder_path)

    def move_working_data_dir(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found")
        os.chdir(self.data_path)
        print(f"Moving to {os.getcwd()}")

    def move_orignal_dir(self):
        if not os.path.exists(self.base_directory):
            raise FileNotFoundError(f"{self.base_directory} not found")
        print(f"Moving back to {os.getcwd()}")
        os.chdir(self.base_directory)

class DataInterface:
    def __init__(self, data_path: str, instance_type: str = "train"):
        self.data_path = data_path
        self.file_manager = FileManager(data_path)
        self.instance_type = instance_type


class DataProcessor(DataInterface):
    def __init__(self, data_path: str, intance_type: str, annotation_filename: str = "_annotations.csv"):
        super().__init__(data_path, intance_type)
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.ground_truth_subpath = "ground_truth"
        self.processed_image_subpath = "processed_img"
        self.numpy_dataset_subpath = "numpy_dataset"
        self.annotation_file = annotation_filename

    def _build_file_achitecture(self) -> None:
        self.file_manager.reset_folder(self.processed_image_subpath)
        self.file_manager.reset_folder(self.numpy_dataset_subpath)
        self.file_manager.reset_folder(self.ground_truth_subpath)

    def _save_ground_truth_images(
            self,
            current_image: np.ndarray,
            ground_truth_images: list[GroundTruthBB],
            image_idx: int
    ) -> None:
        for airplaine_idx, airplaine in enumerate(ground_truth_images):
            filename = f"{image_idx}-{airplaine_idx}.jpg"
            cv2.imwrite(
                f"{self.ground_truth_subpath}/{filename}",
                current_image[airplaine.y1:airplaine.y2, airplaine.x1:airplaine.x2]
            )

    def _get_and_save_selective_search_result(
            self,
            ground_truth_images: list[GroundTruthBB],
            current_image: np.ndarray,
            image_idx: int
    ) -> pd.DataFrame:
        image_df_output = pd.DataFrame(columns=["filename", "x1", "x2", "y1", "y2", "iou", "classname"])
        self.ss.setBaseImage(current_image)
        self.ss.switchToSelectiveSearchFast()
        results = self.ss.process()
        background_found = 0
        features_found = 0
        for res_idx, result in enumerate(results):
            x, y, w, h = result
            bb = SelectiveSBB(x, y, x + w, y + h)
            max_iou = 0
            obj = None
            for gt in ground_truth_images:
                obj = {
                    "x1": bb.x1,
                    "y1": bb.y1,
                    "x2": bb.x2,
                    "y2": bb.y2,
                }
                iou = process_iou(gt, bb)
                max_iou = max(iou, max_iou)
                if iou > 0.5:
                    obj["iou"] = iou
                    obj["classname"] = gt.classname
                    break
            if obj is not None and not obj.get("iou") and background_found > 10:
                continue
            if obj is not None and not obj.get("iou"):
                obj["iou"] = max_iou
                obj["classname"] = class_name_to_class_num.get("background")
                background_found += 1
            if obj is not None and obj.get("iou"):
                if features_found > 20:
                    continue
                features_found += 1
            if obj:
                filename = f"{image_idx}-{res_idx}.png"
                obj["filename"] = filename
                cv2.imwrite(f"{self.processed_image_subpath}/{filename}", current_image[bb.y1:bb.y2, bb.x1:bb.x2])
                image_df_output = pd.concat([image_df_output, pd.DataFrame([obj])])
        return image_df_output

    def _process_images(self):
        if not os.path.exists(f"{self.annotation_file}"):
            raise FileNotFoundError(f"No annotation file found <{self.data_path}>, skipping...")

        annotation_df = pd.read_csv(f"{self.annotation_file}")
        df_output = pd.DataFrame(columns=["filename", "x1", "x2", "y1", "y2", "iou", "classname"])

        num_images = len(os.listdir()) - 1

        for idx, filename in enumerate(os.listdir()):
            if filename == f"{self.annotation_file}" or not filename.startswith("airport"):
                idx -= 1
                continue
            print(f"Process image [{idx} / {num_images}], filename [{filename}]")
            gt_list = []
            sub_df = annotation_df[annotation_df["filename"] == filename]
            current_image = cv2.imread(filename)
            for _, row in sub_df.iterrows():
                _, __, ___, obj_class, x1, y1, x2, y2 = row
                gt_list.append(GroundTruthBB(x1, y1, x2, y2, class_name_to_class_num.get(obj_class)))

            self._save_ground_truth_images(current_image, gt_list, idx)

            obj_annotation_df = self._get_and_save_selective_search_result(gt_list, current_image, idx)
            df_output = pd.concat([df_output, obj_annotation_df])

        df_output.to_csv(f"{self.processed_image_subpath}/{self.annotation_file}")

    def _transorm_dataset_to_array(self, is_ground_truth: bool=False):
        if is_ground_truth:
            path = f"{self.ground_truth_subpath}"
            output_filename = f"{self.ground_truth_subpath}-{self.instance_type}"
        else:
            path = f"{self.processed_image_subpath}"
            annotation_df = pd.read_csv(f"{path}/{self.annotation_file}")
            output_filename = f"{self.processed_image_subpath}-{self.instance_type}"
        images = []
        labels = []
        number_of_pics = len(os.listdir(path))
        for idx, filename in enumerate(os.listdir(path)):
            if filename == f"{self.annotation_file}":
                idx -= 1
                continue
            print(f"image [{idx} / {number_of_pics}]")
            img = cv2.imread(f"{path}/{filename}")
            img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            processed_image = tf.keras.applications.vgg16.preprocess_input(img_resized)
            images.append(processed_image)
            if is_ground_truth:
                labels.append(class_name_to_class_num.get("airplane"))
            else:
                labels.append(annotation_df[annotation_df["filename"] == filename]['classname'])
        np.savez(f"{self.numpy_dataset_subpath}/{output_filename}.npz", images=np.array(images), labels=np.array(labels))

    def run(self):
        print(f"Starting transformation for {self.instance_type}")
        print(f"Move to the datapath <{self.data_path}>, and build the file achitecture...")
        self.file_manager.move_working_data_dir()
        self._build_file_achitecture()

        print("Start processing data...")
        self._process_images()
        self._transorm_dataset_to_array()
        self._transorm_dataset_to_array(is_ground_truth=True)

        print("End of the processing...")
        self.file_manager.move_orignal_dir()


def process_and_create_datasets_from_row_images(train: bool=True, valid: bool=True, test: bool=True):
    data_processors = []
    if train:
        data_processors.append(DataProcessor("data/train", intance_type="train"))
    if valid:
        data_processors.append(DataProcessor("data/valid", intance_type="valid"))
    if test:
        data_processors.append(DataProcessor("data/test", intance_type="test"))

    for processor in data_processors:
        processor.run()


if __name__ == "__main__":
    process_and_create_datasets_from_row_images()
