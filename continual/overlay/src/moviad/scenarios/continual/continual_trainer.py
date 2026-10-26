"""Continual trainer: trains task by task and evaluates every seen task after each one.

Fills the T x T matrix of each metric, then the final average and the forgetting.
"""

import torch
import numpy as np

from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import Metric

class ContinualTrainer:

    def __init__(self,
                 continual_dataset: ContinualDataset,
                 continual_model: ContinualModel,
                 device: torch.device,
                 metrics: list[Metric],
                 training_args: TrainingArgs,
                 logger: any = None
            ):
        """
        Args:
            continual_dataset (ContinualDataset): continual dataset to be used for training
            model (nn.Module): model to be trained
            trainer_arguments (TrainerArguments): arguments for the trainer
        """
        self.continual_dataset = continual_dataset
        self.continual_model = continual_model
        self.trainer_arguments = training_args
        self.metrics = metrics
        self.device = device
        self.logger = logger


    def train(self):

        num_tasks = len(self.continual_dataset)

        # T x T matrix per metric (NaN = not evaluated yet)
        metric_names = [metric.name for metric in self.metrics]
        performance_matrices = {m: np.full((num_tasks, num_tasks), np.nan) for m in metric_names}

        for task_index in range(num_tasks):

            self.continual_model.train()

            print(f"\n--- Training for task: {task_index} , {self.continual_dataset.get_task_category(task_index)} ---")

            train_dataset, eval_dataset = self.continual_dataset.get_task_data(task_index)

            self.continual_model.start_task(task_index, train_dataset, self.trainer_arguments)

            self.continual_model.train_task(
                task_index=task_index,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                train_args=self.trainer_arguments,
                metrics=self.metrics,
                device=self.device,
                logger=self.logger,
            )

            self.continual_model.end_task(task_index, train_dataset, self.trainer_arguments)

            # Evaluate on ALL seen tasks (da 0 fino a task_index INCLUSO)
            summary_metrics = { metric_name: [] for metric_name in metric_names }
            step_logs = {} # logs of this step

            # evaluate every task seen so far
            for eval_task_index in range(task_index + 1):

                eval_dataset = self.continual_dataset.get_task_data_evaluation(eval_task_index)
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=self.trainer_arguments.batch_size,
                    shuffle=False,
                    num_workers=4
                )

                results = Evaluator.evaluate(self.continual_model, eval_dataloader, self.metrics, device=self.device)

                print(f"Performances on task {eval_task_index} after training on task {task_index}:")

                # Salviamo i risultati nella matrice e aggiorniamo le metriche summary e logs
                for metric_name in metric_names:
                    val = results[metric_name]

                    # row = trained task, column = evaluated task
                    performance_matrices[metric_name][task_index, eval_task_index] = val

                    # Aggiorniamo summary
                    summary_metrics[metric_name].append(val)

                    # one log entry per evaluated task
                    prefix = f"Eval_on_Task_{eval_task_index}"
                    step_logs[f"{prefix}/{metric_name}"] = val

                    print(f"  - {metric_name}: {val:.4f}")

            print(f"\nSummary metrics after training on task {task_index}:")
            for metric_name, values in summary_metrics.items():
                avg_value = sum(values) / len(values)
                summary_metrics[metric_name] = avg_value
                print(f"Average {metric_name}: {avg_value:.4f}")

                if self.logger:
                    step_logs[f"Summary_Average/{metric_name}"] = avg_value

            # log the whole step
            if self.logger:
                self.logger.log(step_logs)


        # =========================================================================
        # average forgetting at the end of the sequence
        # =========================================================================
        print("\n" + "="*50)
        print("CALCOLO AVERAGE FORGETTING FINALE")
        print("="*50)

        final_forgetting_logs = {}

        for metric_name in metric_names:
            matrix = performance_matrices[metric_name]
            forgetting_per_task = []

            # the last task cannot be forgotten yet
            for j in range(num_tasks - 1):
                # best value ever on task j
                max_past_perf = np.nanmax(matrix[j:num_tasks-1, j])

                # final value on task j
                final_perf = matrix[num_tasks-1, j]

                forgetting = max_past_perf - final_perf
                forgetting_per_task.append(forgetting)

            # mean over tasks 0..T-2
            avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0
            print(f"Average Forgetting per {metric_name}: {avg_forgetting:.4f}")

            if self.logger:
                final_forgetting_logs[f"Final_Forgetting/{metric_name}"] = avg_forgetting

        if self.logger:
            self.logger.log(final_forgetting_logs)

        # =========================================================================
        # MATRICI T x T + SINTESI FINALE (average finale = media ultima riga)
        # =========================================================================
        categories = [self.continual_dataset.get_task_category(i) for i in range(num_tasks)]
        self.summary = {"categories": categories, "matrices": {}, "final_average": {}, "forgetting": {}}

        for metric_name in metric_names:
            matrix = performance_matrices[metric_name]
            print(f"\nMatrice {metric_name} (riga = dopo training su task i, colonna = valutato su task j):")
            print("        " + " ".join(f"{c[:7]:>7}" for c in categories))
            for i in range(num_tasks):
                row = " ".join("      -" if np.isnan(v) else f"{v:7.4f}" for v in matrix[i])
                print(f"{categories[i][:7]:>7} {row}")

            final_avg = float(np.nanmean(matrix[num_tasks - 1]))
            forgetting = [float(np.nanmax(matrix[j:num_tasks - 1, j]) - matrix[num_tasks - 1, j]) for j in range(num_tasks - 1)]
            self.summary["matrices"][metric_name] = np.where(np.isnan(matrix), None, matrix).tolist()
            self.summary["final_average"][metric_name] = final_avg
            self.summary["forgetting"][metric_name] = float(np.mean(forgetting)) if forgetting else 0.0

        print("\n" + "=" * 50)
        print("SINTESI FINALE")
        print("=" * 50)
        for metric_name in metric_names:
            print(f"{metric_name:>12}: final_avg={self.summary['final_average'][metric_name]:.4f}  "
                  f"forgetting={self.summary['forgetting'][metric_name]:.4f}")

        return performance_matrices
