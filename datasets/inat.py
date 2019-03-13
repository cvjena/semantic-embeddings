import json
import os

from .common import FileDatasetGenerator


# Pre-computed mean and standard deviation for all super-categories
SUPERCATEGORY_STATS = {
    None             : ([119.99310088, 122.86333725, 102.38318464], [60.83471124, 59.33123704, 65.92057842]),
    'actinopterygii' : ([95.60659929, 109.21340134, 99.53273934], [62.64981594, 56.77583425, 57.79043402]),
    'amphibia'       : ([120.38820316, 112.09448704, 93.57291079], [64.38971069, 60.88945117, 60.689195]),
    'animalia'       : ([117.86148813, 112.27558493, 100.76823038], [65.10786879, 60.9941875, 61.3212783]),
    'arachnida'      : ([123.05328454, 123.11786486, 99.49669769], [62.10607939, 59.69295922, 64.12102046]),
    'aves'           : ([125.68554284, 131.58931007, 123.51576605], [56.91926625, 57.04151665, 67.97284604]),
    'bacteria'       : ([130.44253929, 118.58949652, 100.64353881], [63.52655078, 61.3866035, 62.52496727]),
    'chromista'      : ([126.63609004, 120.30744082, 103.69842308], [61.3142875, 60.35121831, 64.33445667]),
    'fungi'          : ([105.4904181, 98.20844854, 81.95195412], [66.43803547, 63.26916273, 61.75505097]),
    'insecta'        : ([126.79141945, 126.55725101, 94.4626541], [62.46710552, 59.70656548, 64.38703598]),
    'mammalia'       : ([119.32537707, 119.28610021, 105.22655576], [60.25561291, 58.86410094, 60.85549787]),
    'mollusca'       : ([119.15865454, 107.82338741, 93.65438902], [65.54171188, 62.00986655, 62.64830566]),
    'plantae'        : ([109.4558912, 115.78290918, 84.83970548], [60.36177593, 59.17162815, 60.81183456]),
    'protozoa'       : ([99.4855571, 90.12976005, 71.67906874], [69.23439903, 63.83415135, 59.1059619]),
    'reptilia'       : ([126.42469824, 119.44987437, 103.84680809], [63.4749642, 60.19704406, 60.20556052]),
}


class INatGenerator(FileDatasetGenerator):

    def __init__(self, root_dir, supercategory=None,
                 cropsize = (224, 224), default_target_size = 256,
                 mean=None, std=None,
                 *args, **kwargs):
        
        super(INatGenerator, self).__init__(root_dir, cropsize=cropsize, default_target_size=default_target_size, *args, **kwargs)

        train_file = os.path.join(root_dir, "train2018.json")
        test_file = os.path.join(root_dir, "val2018.json")

        self.train_tuples, class_count, class_mapping = self.get_tuples_for_supercategory(
            train_file,
            root_dir,
            supercategory=supercategory
        )
        self.test_tuples, class_count, _ = self.get_tuples_for_supercategory(
            test_file,
            root_dir,
            supercategory=supercategory
        )

        self._train_labels, self.train_img_files = zip(*self.train_tuples)
        self._test_labels, self.test_img_files = zip(*self.test_tuples)

        # Set classes
        self.classes = [c for c, idx in sorted(class_mapping.items(), key=lambda t: t[1])]
        self.class_indices = class_mapping

        print('Found {} training and {} validation images from {} classes.'.format(len(self.train_tuples), len(self.test_tuples), class_count))

        # Compute mean and standard deviation
        if (mean is None) and (std is None) and (supercategory in SUPERCATEGORY_STATS):
            mean, std = SUPERCATEGORY_STATS[supercategory]
        self._compute_stats(mean, std)


    def get_tuples_for_supercategory(self, fname, image_folder, supercategory=None):
        """
        Collects the names of the images defined in the provided dataset file and their corresponding class and returns
        them in a list of tuples.

        :param fname: The path to the dataset file.
        :param image_folder: The folder containing the dataset and images in subfolders.
        :param supercategory: The name of the supercategory to use. Possible values are { Chromista, Insecta, Mammalia,
        Arachnida, Aves, Plantae, Fungi, Bacteria, Animalia, Reptilia, Mollusca, Actinopterygii, Protozoa, Amphibia }.
        :return:A list of tuples for each image like [(<class_id>, <filepath>), ...].
        """

        if supercategory is not None:
            supercategory = supercategory.lower()

        with open(fname, "r") as f:
            data = json.loads(f.read())

        id_to_image = { image["id"]: image for image in data["images"] }
        id_to_category = { category["id"]: category for category in data["categories"] if (category["supercategory"].lower() == supercategory) or (supercategory is None) }

        # Create a mapping from the old category ids to new ones that start at 0
        category_id_old_to_new = {id_old: id_new for id_new, id_old in enumerate(sorted(id_to_category.keys()))}

        # Create a class mapping from the string categories to the new ids
        class_mapping = {id_to_category[id_old]["name"]: id_new for id_new, id_old in enumerate(sorted(id_to_category.keys()))}

        valid_annotations = []

        for annotation in data["annotations"]:
            image_id, category_id = annotation["image_id"], annotation["category_id"]

            # Check, if category is in supercategory (id should be in id_to_category dict)
            if category_id in id_to_category:
                filename_abs = os.path.abspath(os.path.join(image_folder, id_to_image[image_id]["file_name"]))

                valid_annotations.append((category_id_old_to_new[category_id], filename_abs))

        return valid_annotations, len(category_id_old_to_new), class_mapping
