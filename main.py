from src.config import load_experiment
from src.experiment import Experiment
from src.renderer import render_empty

def main():
    print("Loading experiment configuration...")
    experiment = load_experiment("config.toml")
    print("Experiment configuration loaded successfully.")
    render_empty(experiment)


if __name__ == "__main__":
    main()
