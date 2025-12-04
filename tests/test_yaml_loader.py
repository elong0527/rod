# pyre-strict
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from tlfyaml.yaml_loader import YamlInheritanceLoader


class TestYamlLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = YamlInheritanceLoader(base_path=Path("/fake/path"))

    def test_load_file_not_found(self) -> None:
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.loader.load("nonexistent.yaml")

    def test_load_simple_yaml(self) -> None:
        yaml_content = "key: value"
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                data = self.loader.load("test.yaml")
                self.assertEqual(data, {"key": "value"})

    def test_resolve_inheritance_no_template(self) -> None:
        data = {"key": "value"}
        resolved = self.loader._resolve_inheritance(data)
        self.assertEqual(resolved, data)

    def test_resolve_inheritance_with_template(self) -> None:
        # Mock loading of template
        template_data = {"common": "data", "override": "old"}

        with patch.object(self.loader, "load", return_value=template_data) as mock_load:
            data = {"study": {"template": "template.yaml"}, "override": "new", "specific": "value"}

            resolved = self.loader._resolve_inheritance(data)

            mock_load.assert_called_once_with("template.yaml")
            self.assertEqual(resolved["common"], "data")
            self.assertEqual(resolved["override"], "new")
            self.assertEqual(resolved["specific"], "value")
            self.assertEqual(resolved["study"]["template"], "template.yaml")

    def test_deep_merge_dicts(self) -> None:
        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {"b": {"d": 3}, "e": 4}

        merged = self.loader._deep_merge(dict1, dict2)

        self.assertEqual(merged, {"a": 1, "b": {"c": 2, "d": 3}, "e": 4})

    def test_deep_merge_lists_simple(self) -> None:
        dict1 = {"l": [1, 2]}
        dict2 = {"l": [2, 3]}

        merged = self.loader._deep_merge(dict1, dict2)

        # Simple list merge: concatenate and dedup
        self.assertEqual(sorted(merged["l"]), [1, 2, 3])

    def test_deep_merge_lists_keyword_dicts(self) -> None:
        # Special handling for lists of dicts with 'name' key
        dict1 = {"items": [{"name": "A", "val": 1}, {"name": "B", "val": 2}]}
        dict2 = {"items": [{"name": "A", "val": 10}, {"name": "C", "val": 3}]}

        merged = self.loader._deep_merge(dict1, dict2)

        # Should merge item A, keep B, add C
        items = {item["name"]: item for item in merged["items"]}
        self.assertEqual(items["A"]["val"], 10)
        self.assertEqual(items["B"]["val"], 2)
        self.assertEqual(items["C"]["val"], 3)
        self.assertEqual(len(items), 3)
