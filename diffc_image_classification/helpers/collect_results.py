from ezcolorlog import root_logger as logger
import os
import json


class ResultsCollector:
    def __init__(self):
        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        self.test_loss = []

    def update_results(
        self,
        epoch,
        train_acc=None,
        train_loss=None,
        test_acc=None,
        test_loss=None,
        lr=None,
    ):
        if train_acc is not None:
            self.train_acc.append(train_acc)
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if test_acc is not None:
            self.test_acc.append(test_acc)
        if test_loss is not None:
            self.test_loss.append(test_loss)

    def print_results(self, epoch, optimizer, scheduler):
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch: {epoch}, LR: {lr}, Train Acc: {self.train_acc[-1]}, Train Loss: {self.train_loss[-1]}, Test Acc: {self.test_acc[-1]}, Test Loss: {self.test_loss[-1]}"
        )

    def save_results(self, results_dir):
        results = {
            "train_acc": self.train_acc,
            "train_loss": self.train_loss,
            "test_acc": self.test_acc,
            "test_loss": self.test_loss,
        }
        results_path = os.path.join(results_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f)
        logger.info(f"Results saved to {results_path}")
