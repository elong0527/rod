# pyre-strict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class YamlInheritanceLoader:
    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.base_path: Path = base_path or Path(".")

    def load(self, file_name: str) -> Dict[str, Any]:
        """
        Load a YAML file by name relative to base_path and resolve inheritance.
        """
        file_path = self.base_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")

        with open(file_path, "r") as f:
            data = yaml.safe_load(f) or {}

        return self._resolve_inheritance(data)

    def _resolve_inheritance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        templates = data.get("study", {}).get("template", [])
        if isinstance(templates, str):
            templates = [templates]

        if not templates:
            return data

        merged_template_data: Dict[str, Any] = {}
        for template_file in templates:
            template_data = self.load(template_file)
            merged_template_data = self._deep_merge(merged_template_data, template_data)

        return self._deep_merge(merged_template_data, data)

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(dict1)
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], list) and isinstance(value, list):
                # Heuristic to check if these are lists of keywords (dicts with a 'name')
                # This logic is specific to how this project uses YAML inheritance.
                is_keyword_list = all(isinstance(i, dict) and "name" in i for i in value) and all(
                    isinstance(i, dict) and "name" in i for i in merged[key]
                )

                if is_keyword_list:
                    merged_by_name = {item["name"]: item for item in merged[key]}
                    for item in value:
                        if item["name"] in merged_by_name:
                            # It's a dict merge, so we can recursively call _deep_merge
                            merged_by_name[item["name"]] = self._deep_merge(
                                merged_by_name[item["name"]], item
                            )
                        else:
                            merged_by_name[item["name"]] = item
                    merged[key] = list(merged_by_name.values())
                else:
                    # Fallback for simple lists: concatenate and remove duplicates
                    # Note: This is a simple approach and might not be suitable for all list types.
                    merged[key].extend([item for item in value if item not in merged[key]])

            elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
