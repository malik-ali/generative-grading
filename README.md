
# Generative Grading

Repository for code related to inference using rubric sampling.

Ideas
- Train RNN/Transformer network on samples of generated code
- Inference directly on rubric sample tree (MCTS, fuzzy matching)


## Directory Structure

The followin directory structure is used throughout the project. In summary, `data/raw` contains raw data which, once preprocessed, is stored in `data/processed`. PyTorch nn.module models are in `src/models` and training agents that use different models and datasets are in `src/agents`.

When a program is run, it loads an experiment specification from a given json file in configs and creates a respective directory for this exepriment in the experiments folder. This stores summaries, logs, checpoints etc.
```
├── configs 				# all experiments are a product of different config files
│   └── rubricrnn.json
├── data
│   ├── processed
│   └── raw
├── experiments
│   └── rubric-rnn
├── scripts
│   ├── main.py 			# Entry point for project
│   └── process_data.py 	# preprocess data
└── src
    ├── agents				# Training agents
    ├── datasets			# PyTorch dataset definitions
    ├── losses				# Different training losses
    ├── models				# Models used for training
    └── utils				# Useful utility methods

```

## Code Usage

To use the code, firstly create a virtual environment (python3) and run `pip install -r requirements.txt`. When you activate the virtual environment, run `source init_env.sh` to set the python classpath.

Put raw data for a problem in `data/raw/[problem-name]`. For example, we might have `data/raw/mirror/simulationCounts_v1_1M_counts.pkl`and`data/raw/mirror/simulationCounts_v1_1M_labels.pkl`.

To preprocess the data, use `scripts/process_data.py [problem-name]`. This will store the preprocessed data in `data/processed/[problem-name]`.

The main way to run the models is to use `python scripts/main.py configs/[your-experiment-config.json]`. The experiment config json contains all relevant information needed to train the model, including which training agent from `agents/`should be used.
