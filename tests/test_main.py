import unittest

from main import infer_default_user_id


class MainUserIdTests(unittest.TestCase):
    def test_infer_default_user_id_returns_existing_user(self):
        user_id = infer_default_user_id("data/ml-100k")

        self.assertIsInstance(user_id, int)
        self.assertGreaterEqual(user_id, 1)


if __name__ == "__main__":
    unittest.main()
