import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable
from tqdm import tqdm

from domain_adaptator import ReverseLayerF
from models import RandomLayer, binary_cross_entropy, cross_entropy_logits, entropy_logits

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common_metrics import classification_metrics, select_threshold_by_f1


class Trainer(object):
    def __init__(
        self,
        model,
        optim,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        opt_da=None,
        discriminator=None,
        experiment=None,
        alpha=1,
        **config,
    ):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.early_stop_patience = config["SOLVER"]["EARLY_STOP_PATIENCE"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(
                    in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                    out_features=config["DA"]["RANDOM_DIM"],
                    bias=False,
                ).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer(
                    [config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"]
                )
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_val_metrics = None
        self.best_threshold = 0.5
        self.best_selection_score = float("-inf")
        self.no_improve_counter = 0
        self.early_stop_epoch = None

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch = []
        self.val_auroc_epoch = []
        self.val_aupr_epoch = []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss", "Threshold"]
        test_metric_header = [
            "# Best Epoch",
            "AUROC",
            "AUPRC",
            "F1",
            "Sensitivity",
            "Specificity",
            "Accuracy",
            "Threshold",
            "Test_loss",
        ]
        if not self.is_da:
            train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (non_init_epoch * self.nb_training)
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def _predict(self, dataloader="test", model_override=None):
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")

        model_ref = model_override if model_override is not None else self.model
        y_label, y_pred = [], []
        loss_total = 0.0
        num_batches = len(data_loader)

        with torch.no_grad():
            model_ref.eval()
            for v_d, v_p, labels in data_loader:
                v_d = v_d.to(self.device)
                v_p = v_p.to(self.device)
                labels = labels.float().to(self.device)
                _, _, _, score = model_ref(v_d, v_p)
                if self.n_class == 1:
                    preds, loss = binary_cross_entropy(score, labels)
                else:
                    preds, loss = cross_entropy_logits(score, labels)
                loss_total += float(loss.item())
                y_label.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

        return y_label, y_pred, loss_total / max(num_batches, 1)

    def _evaluate(self, dataloader="val", threshold=None, model_override=None):
        labels, scores, avg_loss = self._predict(dataloader=dataloader, model_override=model_override)
        if threshold is None:
            threshold = select_threshold_by_f1(labels, scores)
        metrics = classification_metrics(labels, scores, threshold)
        metrics["loss"] = avg_loss
        return metrics

    def _update_best(self, val_metrics):
        selection_score = val_metrics["aupr"]
        improved = selection_score > self.best_selection_score
        if improved:
            self.best_selection_score = selection_score
            self.best_model = copy.deepcopy(self.model)
            self.best_epoch = self.current_epoch
            self.best_val_metrics = dict(val_metrics)
            self.best_threshold = float(val_metrics["threshold"])
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1
            if self.no_improve_counter >= self.early_stop_patience and self.early_stop_epoch is None:
                self.early_stop_epoch = self.current_epoch
        return improved

    def train(self):
        float2str = lambda x: "%0.4f" % x
        for _ in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(
                    map(float2str, [train_loss, model_loss, epoch_lamb, da_loss])
                )
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            val_metrics = self._evaluate(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_metrics["loss"], epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", val_metrics["auc"], epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", val_metrics["aupr"], epoch=self.current_epoch)

            val_lst = ["epoch " + str(self.current_epoch)] + list(
                map(
                    float2str,
                    [
                        val_metrics["auc"],
                        val_metrics["aupr"],
                        val_metrics["loss"],
                        val_metrics["threshold"],
                    ],
                )
            )
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_metrics["loss"])
            self.val_auroc_epoch.append(val_metrics["auc"])
            self.val_aupr_epoch.append(val_metrics["aupr"])
            self._update_best(val_metrics)
            print(
                "Validation at Epoch "
                + str(self.current_epoch)
                + " with validation loss "
                + str(val_metrics["loss"])
                + " AUROC "
                + str(val_metrics["auc"])
                + " AUPRC "
                + str(val_metrics["aupr"])
                + " threshold "
                + str(val_metrics["threshold"])
            )
            if self.early_stop_epoch is not None:
                print(f"Early stopping at epoch {self.early_stop_epoch}")
                break

        if self.best_model is None:
            self.best_model = copy.deepcopy(self.model)
            self.best_epoch = self.current_epoch
            self.best_val_metrics = self._evaluate(dataloader="val")
            self.best_threshold = float(self.best_val_metrics["threshold"])

        test_metrics = self._evaluate(
            dataloader="test",
            threshold=self.best_threshold,
            model_override=self.best_model,
        )
        test_lst = ["epoch " + str(self.best_epoch)] + list(
            map(
                float2str,
                [
                    test_metrics["auc"],
                    test_metrics["aupr"],
                    test_metrics["f1"],
                    test_metrics["recall"],
                    test_metrics["specificity"],
                    test_metrics["acc"],
                    test_metrics["threshold"],
                    test_metrics["loss"],
                ],
            )
        )
        self.test_table.add_row(test_lst)
        print(
            "Test at Best Model of Epoch "
            + str(self.best_epoch)
            + " with test loss "
            + str(test_metrics["loss"])
            + " AUROC "
            + str(test_metrics["auc"])
            + " AUPRC "
            + str(test_metrics["aupr"])
            + " Sensitivity "
            + str(test_metrics["recall"])
            + " Specificity "
            + str(test_metrics["specificity"])
            + " Accuracy "
            + str(test_metrics["acc"])
            + " Threshold "
            + str(test_metrics["threshold"])
        )
        self.test_metrics = {
            "auroc": test_metrics["auc"],
            "auprc": test_metrics["aupr"],
            "test_loss": test_metrics["loss"],
            "sensitivity": test_metrics["recall"],
            "specificity": test_metrics["specificity"],
            "accuracy": test_metrics["acc"],
            "threshold": test_metrics["threshold"],
            "thred_optim": test_metrics["threshold"],
            "best_epoch": self.best_epoch,
            "early_stop_epoch": self.early_stop_epoch,
            "F1": test_metrics["f1"],
            "Precision": test_metrics["precision"],
            "selection_metric": "val_aupr",
            "threshold_policy": "val_f1_optimal",
            "val_metrics": self.best_val_metrics,
        }
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_aupr", self.best_selection_score)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", self.test_metrics["threshold"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(
                self.best_model.state_dict(),
                os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"),
            )
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "val_epoch_aupr": self.val_aupr_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, "result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, "w") as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, "w") as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0.0
        num_batches = len(self.train_dataloader)
        for v_d, v_p, labels in tqdm(self.train_dataloader):
            self.step += 1
            v_d = v_d.to(self.device)
            v_p = v_p.to(self.device)
            labels = labels.float().to(self.device)
            self.optim.zero_grad()
            _, _, _, score = self.model(v_d, v_p)
            if self.n_class == 1:
                _, loss = binary_cross_entropy(score, labels)
            else:
                _, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / max(num_batches, 1)
        print("Training at Epoch " + str(self.current_epoch) + " with training loss " + str(loss_epoch))
        return loss_epoch

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0.0
        model_loss_epoch = 0.0
        da_loss_epoch = 0.0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)
        num_batches = len(self.train_dataloader)
        for batch_s, batch_t in tqdm(self.train_dataloader):
            self.step += 1
            v_d = batch_s[0].to(self.device)
            v_p = batch_s[1].to(self.device)
            labels = batch_s[2].float().to(self.device)
            v_d_t = batch_t[0].to(self.device)
            v_p_t = batch_t[1].to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            _, _, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                _, model_loss = binary_cross_entropy(score, labels)
            else:
                _, model_loss = cross_entropy_logits(score, labels)
            if self.current_epoch >= self.da_init_epoch:
                _, _, f_t, t_score = self.model(v_d_t, v_p_t)
                if self.da_method != "CDAN":
                    raise ValueError(f"The da method {self.da_method} is not supported")

                reverse_f = ReverseLayerF.apply(f, self.alpha)
                softmax_output = torch.nn.Softmax(dim=1)(score).detach()
                if self.original_random:
                    random_out = self.random_layer.forward([reverse_f, softmax_output])
                    adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                else:
                    feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                    feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                    if self.random_layer:
                        random_out = self.random_layer.forward(feature)
                        adv_output_src_score = self.domain_dmm(random_out)
                    else:
                        adv_output_src_score = self.domain_dmm(feature)

                reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                softmax_output_t = torch.nn.Softmax(dim=1)(t_score).detach()
                if self.original_random:
                    random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                    adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                else:
                    feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                    feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                    if self.random_layer:
                        random_out_t = self.random_layer.forward(feature_t)
                        adv_output_tgt_score = self.domain_dmm(random_out_t)
                    else:
                        adv_output_tgt_score = self.domain_dmm(feature_t)

                if self.use_da_entropy:
                    entropy_src = self._compute_entropy_weights(score)
                    entropy_tgt = self._compute_entropy_weights(t_score)
                    src_weight = entropy_src / torch.sum(entropy_src)
                    tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                else:
                    src_weight = None
                    tgt_weight = None

                _, loss_cdan_src = cross_entropy_logits(
                    adv_output_src_score, torch.zeros(self.batch_size).to(self.device), src_weight
                )
                _, loss_cdan_tgt = cross_entropy_logits(
                    adv_output_tgt_score, torch.ones(self.batch_size).to(self.device), tgt_weight
                )
                da_loss = loss_cdan_src + loss_cdan_tgt
                loss = model_loss + da_loss
            else:
                loss = model_loss
                da_loss = torch.tensor(0.0, device=self.device)

            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            da_loss_epoch += da_loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
                if self.current_epoch >= self.da_init_epoch:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)

        total_loss_epoch = total_loss_epoch / max(num_batches, 1)
        model_loss_epoch = model_loss_epoch / max(num_batches, 1)
        da_loss_epoch = da_loss_epoch / max(num_batches, 1)
        if self.current_epoch < self.da_init_epoch:
            print("Training at Epoch " + str(self.current_epoch) + " with model training loss " + str(total_loss_epoch))
        else:
            print(
                "Training at Epoch "
                + str(self.current_epoch)
                + " model training loss "
                + str(model_loss_epoch)
                + ", da loss "
                + str(da_loss_epoch)
                + ", total training loss "
                + str(total_loss_epoch)
                + ", DA lambda "
                + str(epoch_lamb_da)
            )
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da
