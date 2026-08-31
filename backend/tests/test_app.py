import io
import os
import unittest

from PIL import Image

os.environ["CLASSIFIER_MODE"] = "demo"

from app import allowed_file, app


class ImageClassifierApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    def test_extension_validation(self):
        self.assertTrue(allowed_file("photo.JPEG"))
        self.assertFalse(allowed_file("document.pdf"))

    def test_requires_an_image(self):
        response = self.client.post("/api/classify")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose an image", response.json["error"])

    def test_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/classify", data={"image": (io.BytesIO(b"file"), "file.gif")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("JPG", response.json["error"])

    def test_classifies_valid_image_in_demo_mode(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 140, 40)).save(image_buffer, format="PNG")
        image_buffer.seek(0)

        response = self.client.post(
            "/api/classify", data={"image": (image_buffer, "scene.png")}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["predictions"]), 3)
        self.assertIn("natural", response.json["predictions"][0]["label"])
