<div style="text-align: center">
<h1>KinDEL: DNA-Encoded Library Dataset For Kinase Inhibitors</h1>
</div>

KinDEL is a large DNA-encoded library dataset containing two kinase
targets (DDR1 and MAPK14) for benchmarking machine learning models.

## Usage

### Installation

First create environment with dependencies:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
```

### Benchmarking

Run the following command to train a model:

```bash
pixi shell  # activate the environment
PYTHONPATH=. redun -c kindel/.redun run kindel/run.py train \
    --model <model_name, e.g. xgboost_local> \
    --output-dir out \
    --targets ddr1 mapk14 \
    --splits random disynthon \
    --split-indexes 1 2 3 4 5
```

where `<model_name>` has to start with one of the following prefixes:
* xgboost
* rf
* knn
* dnn
* gin
* compose

### Collecting results

To collect the model performance results after training, you
can use the `results.py` script, providing the path to the
model output files:

```bash
python results.py --model-path [path]
```

### Datasets

All datasets are located in AWS S3 at the URL: `s3://kin-del-2024/data`.
You can preview the data using [42basepairs](https://42basepairs.com/) here: https://42basepairs.com/browse/s3/kin-del-2024

The recommended **training dataset** is stored in the `{target}_1M.parquet`
files, which contain top 1M molecules from the DEL screen that were used to
train ML models used in our benchmark.

Data splits are generated in the `splits/{target}_{random/disynthon}.parquet` files,
and the training/validation/testing datasets can be loaded using the
following code:

```python
from kindel.utils.data import get_training_data

df_train, df_valid, df_test = get_training_data(target, split_index=split_index)
```

The results in the benchmark are calculated for the **held-out testing sets** stored
in the `heldout/{target}_{on/off}dna.csv` files, which contain Kd measurements
for the on- and off-DNA compounds. Using the `in_library` argument you can specify
if only the in-library or the extended heldout set is returned. This data can be
loaded using the following code:

```python
from kindel.utils.data import get_testing_data

data = get_testing_data(target, in_library=True)
print(data['on'])
print(data['off'])
```

The full dataset can be downloaded using the following code:

```python
from kindel.utils.data import download_kindel

df = download_kindel(target)
```

### Data structure

All dataset files contain the following columns:
- `smiles` - the SMILES representation of the molecule
- `molecule_hash` - a molecular hash constructed from the synthons that uniquely identifies the molecule
- `smiles_a` - the SMILES of the synthon A
- `smiles_b` - the SMILES of the synthon B
- `smiles_c` - the SMILES of the synthon C

Some compounds in the heldout set do not contain synthon SMILES
strings and the molecule hash. It means that these compounds
were picked from outside the DEL (external compounds in the
extended set).

Besides the molecular structure information, the heldout datasets
contain the `kd` column with the experimental Kd measurements.
The DEL compounds in the training dataset files additionally
contain the following columns:
- `seq_target_1`, `seq_target_2`, `seq_target_3` - sequence counts of the molecules bound to the target in triplicate
- `seq_matrix_1`, `seq_matrix_2`, `seq_matrix_3` - sequence counts of the molecules bound to the control in triplicate
- `seq_load` - the pre population of the molecule
