# pyre-strict
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from tlfyaml.yaml_loader import YamlInheritanceLoader


@pytest.fixture
def loader():
    return YamlInheritanceLoader(base_path=Path("/fake/path"))

def test_load_file_not_found(loader):
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.yaml")

def test_load_simple_yaml(loader):
    yaml_content = "key: value"
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            data = loader.load("test.yaml")
            assert data == {"key": "value"}

def test_resolve_inheritance_no_template(loader):
    data = {"key": "value"}
    resolved = loader._resolve_inheritance(data)
    assert resolved == data

def test_resolve_inheritance_with_template(loader):
    # Mock loading of template
    template_data = {"common": "data", "override": "old"}
    
    with patch.object(loader, "load", return_value=template_data) as mock_load:
        data = {
            "study": {"template": "template.yaml"},
            "override": "new",
            "specific": "value"
        }
        
        resolved = loader._resolve_inheritance(data)
        
        mock_load.assert_called_once_with("template.yaml")
        assert resolved["common"] == "data"
        assert resolved["override"] == "new"
        assert resolved["specific"] == "value"
        assert resolved["study"]["template"] == "template.yaml"

def test_deep_merge_dicts(loader):
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    
    merged = loader._deep_merge(dict1, dict2)
    
    assert merged == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

def test_deep_merge_lists_simple(loader):
    dict1 = {"l": [1, 2]}
    dict2 = {"l": [2, 3]}
    
    merged = loader._deep_merge(dict1, dict2)
    
    # Simple list merge: concatenate and dedup
    assert sorted(merged["l"]) == [1, 2, 3]

def test_deep_merge_lists_keyword_dicts(loader):
    # Special handling for lists of dicts with 'name' key
    dict1 = {"items": [{"name": "A", "val": 1}, {"name": "B", "val": 2}]}
    dict2 = {"items": [{"name": "A", "val": 10}, {"name": "C", "val": 3}]}
    
    merged = loader._deep_merge(dict1, dict2)
    
    # Should merge item A, keep B, add C
    items = {item["name"]: item for item in merged["items"]}
    assert items["A"]["val"] == 10
    assert items["B"]["val"] == 2
    assert items["C"]["val"] == 3
    assert len(items) == 3
