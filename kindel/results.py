import argparse
import os

import numpy as np
import yaml
from redun import File, Dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    for target in ("mapk14", "ddr1"):
        for split_type in ("random", "disynthon"):
            results_dir = Dir(os.path.join(args.model_path, f"results_{split_type}"))
            if not results_dir.exists():
                print(f"Split {split_type} not found. Skipping.")
                continue
            latex = []
            results = {
                "test": {"rho": [], "tau": [], "rmse": []},
                "all": {"on": {"rho": [], "tau": []}, "off": {"rho": [], "tau": []}},
                "lib": {"on": {"rho": [], "tau": []}, "off": {"rho": [], "tau": []}},
            }
            for split_index in range(1, 6):
                out_file = File(
                    os.path.join(
                        results_dir.path, f"results_metrics_s{split_index}_{target}.yml"
                    )
                )
                if not out_file.exists():
                    continue
                with out_file.open("r") as file:
                    content = yaml.load(file, Loader=yaml.Loader)
                    for test_set in ("test", "all", "lib"):
                        if test_set == "test":
                            results[test_set]["rho"].append(content[test_set]["rho"])
                            results[test_set]["tau"].append(content[test_set]["tau"])
                            results[test_set]["rmse"].append(content[test_set]["rmse"])
                        else:
                            for condition in ("on", "off"):
                                results[test_set][condition]["rho"].append(
                                    content[test_set][condition]["rho"]
                                )
                                results[test_set][condition]["tau"].append(
                                    content[test_set][condition]["tau"]
                                )

            for test_set in ("test", "lib", "all"):
                if test_set == "test":
                    metric = "rmse"
                    scores = [score**2 for score in results[test_set][metric]]
                    mu = np.mean(scores)
                    std = np.std(scores)
                    latex.append(f"{mu:.3f} $\pm$ {std:.3f}")
                else:
                    for condition in ("on", "off"):
                        metric = "rho"
                        mu = -np.mean(results[test_set][condition][metric])
                        std = np.std(results[test_set][condition][metric])
                        latex.append(f"{mu:.3f} $\pm$ {std:.3f}")

            print(" & ".join(latex))
