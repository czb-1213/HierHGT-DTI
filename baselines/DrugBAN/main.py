comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
import json
from models import DrugBAN
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold_drug', 'cold_protein'])
parser.add_argument('--data_root', default='./datasets', type=str,
                    help='root directory containing <dataset>/<split>/train,val,test.csv')
parser.add_argument('--output_dir', default=None, type=str,
                    help='directory to save results; default: baselines/DrugBAN/result')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--epochs', default=None, type=int)
parser.add_argument('--patience', default=None, type=int)
parser.add_argument('--lr', default=None, type=float)
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    if args.seed is not None:
        cfg.SOLVER.SEED = args.seed
    if args.epochs is not None:
        cfg.SOLVER.MAX_EPOCH = args.epochs
    if args.patience is not None:
        cfg.SOLVER.EARLY_STOP_PATIENCE = args.patience
    if args.lr is not None:
        cfg.SOLVER.LR = args.lr
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    base_output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "result")
    cfg.RESULT.OUTPUT_DIR = os.path.join(
        base_output_dir,
        f"{args.data}_{args.split}",
        f"seed{cfg.SOLVER.SEED}",
    )
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = os.path.join(args.data_root, args.data, str(args.split))

    if not cfg.DA.TASK:
        train_path = os.path.join(dataFolder, 'train.csv')
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        train_dataset = DTIDataset(df_train.index.values, df_train)
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)
    else:
        train_source_path = os.path.join(dataFolder, 'source_train.csv')
        train_target_path = os.path.join(dataFolder, 'target_train.csv')
        test_target_path = os.path.join(dataFolder, 'target_test.csv')
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "DA_use": cfg.DA.USE,
            "DA_task": cfg.DA.TASK,
        }
        if cfg.DA.USE:
            da_hyper_params = {
                "DA_init_epoch": cfg.DA.INIT_EPOCH,
                "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                "Random_layer": cfg.DA.RANDOM_LAYER,
                "Original_random": cfg.DA.ORIGINAL_RANDOM,
                "DA_optim_lr": cfg.SOLVER.DA_LR
            }
            hyper_params.update(da_hyper_params)
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    if not cfg.DA.USE:
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        if not cfg.DA.TASK:
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)

    model = DrugBAN(**cfg).to(device)

    if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if not cfg.DA.USE:
        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)
    else:
        trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm,
                          experiment=experiment, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    unified_result = {
        "model": "DrugBAN",
        "dataset": args.data,
        "split": args.split,
        "seed": cfg.SOLVER.SEED,
        "auc": round(float(result["auroc"]), 4),
        "aupr": round(float(result["auprc"]), 4),
        "f1": round(float(result["F1"]), 4),
        "acc": round(float(result["accuracy"]), 4),
        "best_epoch": result.get("best_epoch"),
        "early_stop_epoch": result.get("early_stop_epoch"),
        "test_loss": round(float(result["test_loss"]), 4),
        "threshold": round(float(result["threshold"]), 6),
        "threshold_policy": result.get("threshold_policy"),
        "selection_metric": result.get("selection_metric"),
    }
    json_path = os.path.join(cfg.RESULT.OUTPUT_DIR, "test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(unified_result, f, indent=2)
    print(f"Unified results saved to {json_path}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
