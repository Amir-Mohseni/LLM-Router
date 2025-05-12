import os
import re

def load_env_variables_from_example(env_path=".env.example", ignore_keys=None):
    """
    Loads environment variables from a .env.example file and sets them in os.environ,
    excluding any keys listed in ignore_keys.

    Args:
        env_path (str): Path to the .env.example file.
        ignore_keys (set or list): A set or list of keys to skip (e.g., {'TOKENIZERS_PARALLELISM'}).
    """
    if ignore_keys is None:
        ignore_keys = set()
    else:
        ignore_keys = set(ignore_keys)

    env_var_pattern = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.+?)(?:\s+#.*)?$")

    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {env_path}")
        return

    for line in lines:
        match = env_var_pattern.match(line)
        if match:
            key = match.group(1)
            if key in ignore_keys:
                continue
            raw_value = match.group(2).strip()
            value = raw_value.strip("'\"")
            os.environ[key] = value
            print(f"os.environ[\"{key}\"] = \"{value}\"")

load_env_variables_from_example(ignore_keys={"VLLM_API_KEY"})


from routellm.controller import Controller

class RouteLLMClassifier:
    def __init__(self, strong_model, weak_model):
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.client = Controller(
            routers=["mf"],
            strong_model = self.strong_model,
            weak_model = self.weak_model,
        )

    def classify(self, text):
        response = self.client.chat.completions.create(
            # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
            model="router-mf-0.11593",
            messages=[
                {"role": "user", "content": text}
            ]
        )
        return response
