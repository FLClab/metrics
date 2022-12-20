
import subprocess
import os, shutil
import glob
import platform
import tifffile
import numpy

PLATFORMOS = {
    "Darwin" : "Mac",
    "Linux" : "Linux",
    "Windows" : "Win"
}

class CTCMeasure:
    """
    Implements the Cell Tracking Challenge measures using the provided
    scripts.
    """
    def __init__(self, truths, predictions, num_digits=3, tmp_folder="./tmp", verbose=False):
        """
        Instantiates the `CTCMeasure` metric object

        :param truths: A `list` of ground truth segmentations
        :param predictions: A `list` of predicted segmentations (same order as truths)
        :param num_digits: An `int` of the number of digits to use for the images
        :param tmp_folder: A `str` of the location of copies files
        :param verbose: A `bool` wether metric calculation should be verbosed
        """
        self.truths = truths
        self.predictions = predictions
        self.num_digits = num_digits
        self.tmp_folder = tmp_folder

        self.verbose = None if verbose else subprocess.PIPE

        assert platform.system() in ["Linux", "Darwin"]

    def get_seg(self):
        """
        Computes SEG metric

        :returns : A `float` of the SEG metric
        """
        segmeasure = os.path.join(os.path.dirname(__file__), PLATFORMOS[platform.system()], "SEGMeasure")

        # Launch the process
        p = subprocess.Popen(
            [segmeasure, self.tmp_folder, "01", str(self.num_digits)],
            stdout=self.verbose, stderr=self.verbose
        )
        p.wait()

        # Reads result from file
        with open(os.path.join(self.tmp_folder, "01_RES", "SEG_log.txt"), "r") as file:
            lines = [line for line in file.readlines()]
        return eval(lines[-1].split(":")[-1])

    def get_det(self):
        """
        Computes DET metric

        :returns : A `float` of the DET metric
        """
        detmeasure = os.path.join(os.path.dirname(__file__), PLATFORMOS[platform.system()], "DETMeasure")

        # Launch the process
        p = subprocess.Popen(
            [detmeasure, self.tmp_folder, "01", str(self.num_digits)],
            stdout=self.verbose, stderr=self.verbose
        )
        p.wait()

        # Reads result from file
        with open(os.path.join(self.tmp_folder, "01_RES", "DET_log.txt"), "r") as file:
            lines = [line for line in file.readlines()]
        return eval(lines[-1].split(":")[-1])

    def get_tra(self):
        """
        Computes TRA metric
        """
        raise NotImplementedError("This metric is not implemented... yet!")

    def create_folder(self):
        """
        Creates a temporary folder architecture from the
        given files in truths and predictions
        """
        # Makes sure the temporary folder is empty
        self.remove_folder()

        # Creates tmp folder
        os.makedirs(os.path.join(self.tmp_folder, "01_GT", "SEG"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_folder, "01_GT", "TRA"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_folder, "01_RES"), exist_ok=True)

        # Copies truths and predictions files to tmp folder
        for i, file in enumerate(self.truths):
            idx = str(i).zfill(self.num_digits)
            if isinstance(file, str):
                shutil.copy(file, os.path.join(self.tmp_folder, "01_GT", "SEG", f"man_seg{idx}.tif"))
                shutil.copy(file, os.path.join(self.tmp_folder, "01_GT", "TRA", f"man_track{idx}.tif"))
            else:
                tifffile.imsave(os.path.join(self.tmp_folder, "01_GT", "SEG", f"man_seg{idx}.tif"), file.astype(numpy.uint16))
                tifffile.imsave(os.path.join(self.tmp_folder, "01_GT", "TRA", f"man_track{idx}.tif"), file.astype(numpy.uint16))
        for i, file in enumerate(self.predictions):
            idx = str(i).zfill(self.num_digits)
            if isinstance(file, str):
                shutil.copy(file, os.path.join(self.tmp_folder, "01_RES", f"mask{idx}.tif"))
            else:
                tifffile.imsave(os.path.join(self.tmp_folder, "01_RES", f"mask{idx}.tif"), file.astype(numpy.uint16))

    def remove_folder(self):
        """
        Removes a temporary folder architecture
        """
        # Removes previous folders
        if os.path.isdir(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def __enter__(self):
        # Creates the tmp folder
        self.create_folder()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Deletes the tmp folder
        self.remove_folder()

if __name__ == "__main__":

    truths = list(sorted(glob.glob("/Users/anthony/Downloads/EvaluationSoftware/testing_dataset/03_GT/TRA/*.tif")))
    predictions = list(sorted(glob.glob("/Users/anthony/Downloads/EvaluationSoftware/testing_dataset/03_RES/*.tif")))
    with CTCMeasure(truths, predictions) as ctc_measure:
        seg = ctc_measure.get_seg()
        det = ctc_measure.get_det()
