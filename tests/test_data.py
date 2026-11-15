import unittest

from prepare_data import make_splits


class SplitTests(unittest.TestCase):
    def test_duplicates_cannot_cross_splits_and_test_is_preserved(self):
        records = [{"path": f"train/{label}/{index}.jpg", "label": label,
                    "digest": f"{label}-{index}"} for label in (0, 1) for index in range(20)]
        records += [{"path": "test/REAL/a.jpg", "label": 0, "digest": "0-0"},
                    {"path": "test/REAL/b.jpg", "label": 0, "digest": "0-0"},
                    {"path": "test/FAKE/a.jpg", "label": 1, "digest": "test-fake"},
                    {"path": "train/REAL/duplicate.jpg", "label": 0, "digest": "0-1"}]
        rows, audit = make_splits(records)
        self.assertEqual((rows, audit), make_splits(records))
        self.assertEqual(sum(row["split"] == "test" for row in rows), 3)
        hashes = {split: {row["digest"] for row in rows if row["split"] == split}
                  for split in ("train", "val", "test")}
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(hashes[first] & hashes[second])
        self.assertEqual(audit["training_images_excluded_for_test_overlap"], 1)
        self.assertEqual(audit["additional_training_duplicates_excluded"], 1)
        self.assertEqual(audit["duplicates_within_official_test_retained"], 1)
        for split in ("train", "val", "test"):
            self.assertEqual({row["label"] for row in rows if row["split"] == split}, {0, 1})

    def test_insufficient_unique_data_is_rejected(self):
        with self.assertRaises(ValueError):
            make_splits([])


if __name__ == "__main__":
    unittest.main()
