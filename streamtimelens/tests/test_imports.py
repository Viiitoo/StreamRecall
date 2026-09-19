import unittest


class ImportTest(unittest.TestCase):
    def test_package_import(self):
        import streamtimelens
        self.assertEqual(streamtimelens.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
