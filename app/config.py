import yaml

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Invalid YAML in config file {config_path}: {str(e)}")
    
CONFIG = load_config()