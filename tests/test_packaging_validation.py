"""Submission must reject incomplete hashes and mislabeled source revisions."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("packaging_validation", ROOT / "scripts/validate-packaging.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
PIN = "1868a6273174f58856af25ba6c45eb057d3bbfd0"


class PackagingValidationTests(unittest.TestCase):
    def test_prepared_candidates_pass(self):
        for kind, path in validation.RECIPES.items():
            with self.subTest(kind=kind):
                validation.validate_recipe(kind, (ROOT / path).read_text(), PIN, "0.1.0", "0.1.0")

    def test_future_release_cannot_relabel_old_source(self):
        for kind, path in validation.RECIPES.items():
            text = (ROOT / path).read_text()
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "source differs"):
                    validation.validate_recipe(kind, text, "a" * 40, "0.1.0", "0.1.0")
                with self.assertRaisesRegex(ValueError, "Requested version differs"):
                    validation.validate_recipe(kind, text, PIN, "0.1.0", "1.0.0")
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    validation.validate_recipe(kind, text + "\nREPLACE_WITH_ACTUAL_HASH", PIN, "0.1.0", "0.1.0")

    def test_compressed_tarball_hex_is_not_a_nix_sri_hash(self):
        text = (ROOT / validation.RECIPES["nix"]).read_text()
        import re
        text = re.sub(r'hash = "sha256-[^"]+"', 'hash = "sha256-' + '0' * 64 + '"', text)
        with self.assertRaisesRegex(ValueError, "SRI digest"):
            validation.validate_recipe("nix", text, PIN, "0.1.0", "0.1.0")

    def test_macports_must_include_offline_crates(self):
        text = (ROOT / validation.RECIPES["macports"]).read_text().split("cargo.crates")[0]
        with self.assertRaisesRegex(ValueError, "locked Cargo dependencies"):
            validation.validate_recipe("macports", text, PIN, "0.1.0", "0.1.0")


if __name__ == "__main__":
    unittest.main()
