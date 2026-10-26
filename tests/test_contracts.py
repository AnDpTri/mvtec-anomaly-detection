"""Kiểm tra hợp đồng giao tiếp, không dùng kết quả benchmark thay cho kiểm thử."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from src.autoencoder.model import ConvAutoencoder
from src.demo.artifacts import (
    save_artifact,
    load_artifact,
    prepare_image,
    infer_tensor,
    checkpoint_path,
)
from src.data.mvtec import MVTecDataset
from src.evaluation.metrics import image_auroc, pixel_auroc

torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    def test_known_auroc_and_shape_validation(self):
        self.assertEqual(image_auroc([0, 1], [0, 1]), 1.0)
        self.assertEqual(image_auroc([0, 1], [1, 0]), 0.0)
        with self.assertRaises(ValueError):
            pixel_auroc(np.zeros((2, 8, 8)), np.zeros((2, 4, 4)))

    def test_loader_good_only_and_binary_mask(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for folder in [
                "bottle/train/good",
                "bottle/test/good",
                "bottle/test/crack",
                "bottle/ground_truth/crack",
            ]:
                (root / folder).mkdir(parents=True)
            image = Image.fromarray(np.full((19, 23), 128, np.uint8))
            for name in [
                "bottle/train/good/000.png",
                "bottle/test/good/000.png",
                "bottle/test/crack/001.png",
            ]:
                image.save(root / name)
            mask = np.zeros((19, 23), np.uint8)
            mask[:9, :8] = 255
            Image.fromarray(mask).save(root / "bottle/ground_truth/crack/001_mask.png")
            train = MVTecDataset(root, "bottle", "train", 32)
            test = MVTecDataset(root, "bottle", "test", 32)
            self.assertEqual(tuple(train[0]["image"].shape), (3, 32, 32))
            self.assertEqual(train[0]["label"], 0)
            self.assertEqual(len(test), 2)
            self.assertEqual(set(test[0]["mask"].unique().tolist()), {0.0, 1.0})

    def test_checkpoint_roundtrip_same_inference(self):
        with tempfile.TemporaryDirectory() as root:
            torch.manual_seed(42)
            model = ConvAutoencoder(latent_dim=8, image_size=32).eval()
            config = {
                "latent_dim": 8,
                "image_size": 32,
                "loss": "mse",
                "sigma": 1,
                "threshold": None,
            }
            path = Path(root) / "model.pt"
            save_artifact(model, "ae", "bottle", config, path)
            loaded, c, method = load_artifact(str(path), "cpu")
            x = prepare_image(Image.new("RGB", (43, 37), "gray"), config, "ae", "cpu")
            expected, s1 = infer_tensor(model, x, "ae", config)
            actual, s2 = infer_tensor(loaded, x, method, c)
            np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(s1, s2)
            load_artifact.cache_clear()

    def test_checkpoint_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            checkpoint_path("ae", "../other")


if __name__ == "__main__":
    unittest.main()
