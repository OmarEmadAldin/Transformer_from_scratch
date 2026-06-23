from pathlib import Path
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": "opus-100",
        "lang_src": "en",
        "lang_tgt": "ar",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/experiment_1"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = Path(config["model_folder"])
    if not model_folder.exists():
        return None
    weights_files = list(
        model_folder.glob(f"{config['model_basename']}*.pt")
    )
    if len(weights_files) == 0:
        return None
    weights_files.sort(key=lambda x: x.stat().st_mtime)
    return str(weights_files[-1])