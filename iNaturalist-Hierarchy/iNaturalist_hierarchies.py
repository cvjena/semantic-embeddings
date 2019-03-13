import json


def generate_parent_child_pairs(path, supercategory=None):
    classes = [
        #"supercategory",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "name"
    ]

    with open(path, "r") as f:
        data = json.loads(f.read())

    parent_child_pairs = set()

    for category in data["categories"]:
        if supercategory is None or category["supercategory"] == supercategory:

            # Super-super category to connect all kingdoms into a tree
            parent_child_pairs.add(("__NULL__", category["kingdom"]))

            for i in range(len(classes) - 1):
                parent_child_pairs.add((category[classes[i]], category[classes[i + 1]]))

    for pair in sorted(list(parent_child_pairs)):
        print("{} {}".format(*pair))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This tool generates a parent-child file for semantic embeddings for "
                                                 "the iNaturalist dataset and possibly a selected supercategory.")

    parser.add_argument("dataset_path", type=str, help="The path to the training file of the iNaturalist 2018 dataset.")
    parser.add_argument("--supercategory", type=str, default=None,
                        help="The name of the supercategory the hierarchy will be based on.")

    args = parser.parse_args()

    # dataset_path = "/home/datasets1/inat/2018/train2018.json"
    generate_parent_child_pairs(args.dataset_path, supercategory=args.supercategory)
