import yaml


class APIPricingConfig:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.models = self.load_yaml()

    def load_yaml(self):
        try:
            with open(self.yaml_file, "r") as file:
                data = yaml.safe_load(file)
                return data.get("models", [])
        except FileNotFoundError:
            print(f"Error: The file {self.yaml_file} was not found.")
            return []
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return []

    def get_model_names(self):
        return [model["name"] for model in self.models]

    def get_model_by_name(self, name):
        for model in self.models:
            if model["name"] == name:
                return model
        return None

    def __str__(self):
        return "\n".join(
            f"Name: {model['name']}, Input: {model['input']}, Output: {model['output']}"
            for model in self.models
        )
