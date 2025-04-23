"""
Framework compatibility checker module.

This module provides functionality to check if a framework is compatible with
a specified model family based on the framework_compatibility.yaml configuration.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any


class FrameworkCompatibilityError(Exception):
    """Exception raised when a framework is not compatible with a model family."""
    pass


class CompatibilityChecker:
    """
    Checks compatibility between frameworks and model families.
    """

    def __init__(self):
        """Initialize the compatibility checker with data from YAML file."""
        yaml_path = os.path.join(
            Path(__file__).parent, "framework_compatibility.yaml"
        )
        with open(yaml_path, "r") as file:
            self.compatibility_data = yaml.safe_load(file)

    def is_compatible(self, framework_name: str, model_family: str) -> bool:
        """
        Check if a framework is compatible with a model family.

        Args:
            framework_name: Name of the framework class
            model_family: Model family to check (openai, google, ollama, transformers)

        Returns:
            bool: True if compatible, False otherwise
        """
        if framework_name not in self.compatibility_data:
            # If framework is not in the compatibility list, assume it's unrestricted
            return True

        supported_families = self.compatibility_data[framework_name].get("supported_families", [])
        return model_family in supported_families

    def check_compatibility(self, framework_name: str, model_family: str) -> None:
        """
        Check if a framework is compatible with a model family and raise an error if not.

        Args:
            framework_name: Name of the framework class
            model_family: Model family to check

        Raises:
            FrameworkCompatibilityError: If the framework is not compatible with the model family
        """
        if not self.is_compatible(framework_name, model_family):
            supported = self.compatibility_data[framework_name].get("supported_families", [])
            supported_str = ", ".join(supported)
            raise FrameworkCompatibilityError(
                f"Framework '{framework_name}' is not compatible with model family '{model_family}'. "
                f"Supported model families: {supported_str}"
            )

    def get_supported_families(self, framework_name: str) -> List[str]:
        """
        Get the list of supported model families for a framework.

        Args:
            framework_name: Name of the framework class

        Returns:
            List of supported model family names
        """
        if framework_name not in self.compatibility_data:
            return ["openai", "google", "ollama", "transformers"]  # Default to all
            
        return self.compatibility_data[framework_name].get("supported_families", [])


# Singleton instance for easy import
compatibility_checker = CompatibilityChecker()