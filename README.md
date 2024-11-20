<div style="text-align: center">
<h1>KinDEL: DNA-Encoded Library Dataset For Kinase Inhibitors Cheminformatics Tech Challenge</h1>
</div>

# 1) Introduction
This repository serves as the submission for the KinDEL: DNA-Encoded Library Dataset For Kinase Inhibitors [Cheminformatics Tech Challenge](https://loka.notion.site/Cheminformatics-Tech-Challenge-13046e8c24378037a76df8f028166aff). The critical analysis of the work presented is presented in this `.README` file. This file is organized in the following sections:

- Introduction
- Setup & Installation
- Data & Computational Resources
- Problem Definition & approach
- Results
- Discussion & Future Work 
- Conclusion


# 2) Setup & Installation
In this section, the necessary application and setup steps will be covered. This solution was done in a Windows 10 machine and so all the setup will be explained in that context.

## 2.1) Installation
The following applications need to be installed:
- [git (and the accompanying git bash)](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Python package manager [Poetry version 1.8.4](https://python-poetry.org/docs/#installing-with-pipx)
- [Visual Studio Code](https://code.visualstudio.com/download)

## 2.2) Setup
- The repository should be cloned from [github](https://github.com/ar-neto/kindel)
- Access to S3 buckets is required by repository's models. As such, an AWS access key id and and AWS secret access key need to be [created with this guide](https://stackoverflow.com/questions/21440709/how-do-i-get-aws-access-key-id-for-amazon)
- Then a file named `.env` should be created in the root of the cloned repository's directory
- There, the file should be filed in the following fashion:

```
AWS_ACCESS_KEY_ID=[your AWS secret key ID]
AWS_SECRET_ACCESS_KEY=[youre AWS secret key]
```

- Visual Studio code should be opened, followed by opening a git bash shell . There, the following command is excuted:

```bash
export PIPENV_VENV_IN_PROJECT=1
``` 

- Finally, the environment is created by running:
```bash
poetry install
poetry shell
```

After all these steps, model training can be initiated via:
```bash

poetry run python .venv/Scripts/redun -c kindel/.redun run kindel/run.py train --model moe_local --output-dir model_results --targets ddr1 mapk14 --splits random disynthon --split-indexes 1 2 3 4 5
```

and the subsequent results can be consulted by running the following command:
```bash
python kindel/results.py --model-path model_results/moe_local
```


# 3) Data & Computational Resources

## 3.1) Data
The data utilised in this challenge was the KinDEL dataset, made available in its [Github repository](https://github.com/insitro/kindel). As described in its [accompanying paper](https://arxiv.org/pdf/2410.08938), the dataset contains DNA-encoded library (DEL) data for DDR1 and MAPK14, two kinase targets. 

## 3.2) Computational Resources
This challenge was solved on a personal Windows 10 laptop, with 8 Gb of RAM (Random Access Memory) and a 6 Gb NVIDIA GeForce RTX 3060 Laptop GPU. As such, not all of the data provided was utilised to train this challenge's model.


# 4) Problem Definition & approach

## 4.1) Problem Definition
As [described](https://loka.notion.site/Cheminformatics-Tech-Challenge-13046e8c24378037a76df8f028166aff), this challenge's goal it to build a machine learning model to predict enrichment scores of DEL compounds and evaluate how well it generalizes to real binding affinities (Kd). Consequently, the problem was construed as a molecule binding affinity problem, where the molecules of a dataset, through some specific representation (e.g. [SMILES](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System)), the molecule's binding affinity are learned by a model.

## 4.2) Approach
Now that the problem is being tackled as a protein binding problem, different alternatives can be chosen. The benchmark models provided with the [KinDEL dataset](https://github.com/insitro/kindel) already represent a few viable options. When compared with its [paper]((https://arxiv.org/pdf/2410.08938))'s tables 1 and 2, it becomes apparent that the DEL-Compose approach looks promising as a basis to build upon. However, it is rather computationally expensive, thus making unfeasible. So, from the remaining options, the Deep Neural Network (DNN) approach seems to perform comparatively well while still being scalable enough.

Looking into [literature](https://paperswithcode.com/paper/high-performance-of-gradient-boosting-in), Gradient Boosted Trees seem like yet another promising yet light-weight approach to molecule binding affinity problems. Furthermore,  [some papers](https://www.biorxiv.org/content/10.1101/2024.08.06.606753v1.full) seem to indicate that a Mixture of Experts approach to these problems can further improve the models' predictive power. Furthermore, the DEL-compose's consistent performance in tables 1 and 2 also seems to indicate that come form of combined models improves the outcome.

In light of this information, the approach taken was a Mixture of Experts approach, with two experts. One of the experts was a Neural Network with the following architecture:

```python
nn.Linear(input_size, hidden_size),
nn.BatchNorm1d(hidden_size),
nn.Dropout(p=0.2),
nn.ReLU(),
nn.Linear(hidden_size, hidden_size),
nn.BatchNorm1d(hidden_size),
nn.Dropout(p=0.2),
nn.ReLU(),
nn.Linear(hidden_size, hidden_size),
nn.BatchNorm1d(hidden_size),
nn.Dropout(p=0.2),
nn.ReLU(),
nn.Linear(hidden_size, 1),
```

This architecture was a smaller version of the DNN benchmark provided in the KinDEL repository. THis was done to retain as much model capacity as possible while still minimizing the required computational resources.

The other expert was a Gradient Boosted Tree, as implemented in the KinDEL repository. 

As for the Mixture of Experts portion, its output is a weighted sum of the experts' output. Now, these weights defined by the gating function.


# 5) Results
A results summary can be found below. The full results table can be found in the `results_table.md` file. The general trends observed below are the differences in performance between the split strategies and the data splits.

Table 1: mapk14 target results 
| Target | Split strategy   | Metric                                   | Data split           | Value (mean ± standard deviation) |
| ------ | ---------------- | ---------------------------------------- | -------------------- | --------------------------------- |
| mapk14 | random           | RMSE                                     | test set             | 0.128 ± 0.144                     |
| mapk14 | random           | Spearman Correlation coefficient         |  on DNA:In-library   | 0.410 ± 0.153                     |
| mapk14 | random           | Spearman Correlation coefficient         | off DNA:In-library   | 0.406 ± 0.124                     |
| mapk14 | random           | Kendall's tau                            | off DNA:In-library   | 0.272 ± 0.084                     |
| mapk14 | random           | Spearman Correlation coefficient         |  on DNA:all          | 0.089 ± 0.186                     |
| mapk14 | random           | Spearman Correlation coefficient         | off DNA:all          | 0.418 ± 0.113                     |
| mapk14 | random           | Kendall's tau                            | off DNA:all          | 0.280 ± 0.081                     |
| mapk14 | disynthon        | RMSE                                     | test set             | 0.168 ± 0.059                     |
| mapk14 | disynthon        | Spearman Correlation coefficient         |  on DNA:In-library   | 0.046 ± 0.238                     |
| mapk14 | disynthon        | Spearman Correlation coefficient         | off DNA:In-library   | 0.246 ± 0.055                     |
| mapk14 | disynthon        | Kendall's tau                            | off DNA:In-library   | 0.167 ± 0.043                     |
| mapk14 | disynthon        | Spearman Correlation coefficient         |  on DNA:all          | -0.059 ± 0.171                    |
| mapk14 | disynthon        | Spearman Correlation coefficient         | off DNA:all          | 0.227 ± 0.096                     |
| mapk14 | disynthon        | Kendall's tau                            | off DNA:all          | 0.154 ± 0.073                     |

Table 2: mapk14 target results
| Target | Split strategy   | Metric                                   | Data split           | Value (mean ± standard deviation) |
| ddr1   | random           | RMSE                                     | test set             | 0.530 ± 0.117                     |
| ddr1   | random           | Spearman Correlation coefficient         | on DNA:In-library    | 0.458 ± 0.065                     |
| ddr1   | random           | Spearman Correlation coefficient         | off DNA:In-library   | 0.181 ± 0.092                     |
| ddr1   | random           | Kendall's tau                            | off:DNA:In-library   | 0.126 ± 0.060                     |
| ddr1   | random           | Spearman Correlation coefficient         | on DNA:all           | 0.512 ± 0.049                     |
| ddr1   | random           | Spearman Correlation coefficient         | off:DNA:all          | 0.170 ± 0.082                     |
| ddr1   | random           | Kendall's tau                            | off:DNA:all          | 0.121 ± 0.055                     |
| ddr1   | disynthon        | RMSE                                     | test set             | 1.722 ± 1.061                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | on DNA:In-library    | 0.367 ± 0.214                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | off:DNA:In-library   | 0.090 ± 0.086                     |
| ddr1   | disynthon        | Kendall's tau                            | off:DNA:In-library   | 0.060 ± 0.055                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | on DNA:all           | 0.486 ± 0.137                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | off:DNA:all          | 0.087 ± 0.083                     |
| ddr1   | disynthon        | Kendall's tau                            | off:DNA:all          | 0.059 ± 0.054                     |



6) Discussion & Future Work

6.1) Discussion
The primary challenge encountered during this project was the significant limitation in computational resources. This restriction constrained the amount of data that could be processed within the challenge’s time frame, making it impossible to reproduce all the benchmarks reported in the repository.

As a result, model training was conducted on a random subset comprising only 0.1% of the original dataset. This reduced subset limited the exploration of the problem space, negatively affecting the model’s performance metrics and its ability to generalize effectively.

Analyzing the metrics, it is evident that the results obtained were significantly lower than those presented in the KenDAL paper, as anticipated given the constraints. Examining the results by data split, both targets consistently achieved better performance on the random split, a trend also observed in the paper’s DNN and XGBoost models. This suggests that the Mixture of Experts (MoE) approach preserved this behavior.

When examining performance by metric, the results reveal an interesting pattern. For the ddr1 target, the Spearman Correlation coefficient for -on DNA data was higher than its -off DNA counterpart, whereas for the mapk14 target, the opposite was observed. This seems to indicate that this approach's ability to learn -on and -off DNA approaches depends on the target. Additionally, this could also be caused by the subsampled data's distribution rather than the model's ability to learn different features. For these favourable instances, this model has successfully achieved values comparable to the values in the KenDAL paper, meaning that further computational power could aid these results. 

6.2) Future work
The lack of computational power impose several design constraints (from the model selection to the training data), meaning that investment can widen the model's possibilities.

As for model architecture, different kinds of neural networks should be explored. For instance, [Graph Neural Networks are popular approaches to this problem](https://paperswithcode.com/sota/protein-ligand-affinity-prediction-on-pdbbind). Additionally, there has been work done to utilise [transformer models as well](https://paperswithcode.com/paper/plapt-protein-ligand-binding-affinity).

As for the experts, employing more than 2 experts, given enough computational power, can also aid the prediction, as well as experimenting with other Mixture of Experts approaches.

As for visualization, the models (both the experts, the gating function and the final model) can all be exported to allow for further experimentation without retraining the model, as well as the elaboration of visualizations to further observe the model's behaviour with samples outside of the training data.
