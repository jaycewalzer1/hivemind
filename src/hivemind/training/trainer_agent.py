"""
@file: trainer_agent_pytorch.py
Created on 31.05.22
@project: CrazyAra
@author: queensgambit

Definition of the main training loop done in pytorch.
Partially based on:
https://gitlab.com/jweil/PommerLearn/-/blob/master/pommerlearn/training/train_cnn.py
"""

import random
import logging
import glob
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
from time import time
import datetime
import onnx
from rtpt import RTPT
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch import Tensor
from onnxsim import simplify

from hivemind.inference.onnx_graph import topologically_sort_graph
from hivemind.config.main_config import main_config
from hivemind.config.train_config import TrainConfig, TrainObjects
from hivemind.training.data_loaders import load_parquet_shard
from hivemind.training.train_util import return_metrics_and_stop_training


def evaluation_batch_limit(train_config: TrainConfig, full_evaluation: bool):
    return None if full_evaluation else train_config.eval_batches


def requires_full_evaluation(
    train_config: TrainConfig,
    training_complete: bool,
    epoch_complete: bool,
):
    return training_complete or (
        train_config.full_eval_each_epoch and epoch_complete
    )


def autocast_context(ctx, precision):
    if precision == "fp32" or ctx.type != "cuda":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"Unsupported mixed precision: {precision}")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class TrainerAgentPytorch:
    """Main training loop"""

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        train_config: TrainConfig,
        train_objects: TrainObjects,
        use_rtpt: bool,
        additional_loaders=None,
        is_rl: bool = False,
        rl_train_loader: DataLoader = None,
        train_eval_loader: DataLoader = None,
    ):
        """
        Class for training the neural network.
        :param net: The NN with loaded parameters that shall be trained.
        :param val_data: The validation data loaded with gluon DataLoader.
        :param train_config: An instance of the TrainConfig data class.
        :param train_objects: Am instance pf the TrainObject data class.
        :param use_rtpt: If True, an RTPT object will be created and modified within this class.
        :param additional_loaders: optional dictionary of {dataset_name: DataLoader} whose dataloaders will also be
         used for evaluation (only used for informative purposes)
        :param is_rl: If True, use RL training mode with soft policy targets
        :param rl_train_loader: DataLoader for RL training data (used when is_rl=True)
        """
        self.additional_loaders = additional_loaders
        self.tc = train_config
        self.to = train_objects
        self.weights_dir = (
            Path(self.tc.weights_dir)
            if self.tc.weights_dir
            else Path(self.tc.export_dir) / "weights"
        )
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.is_rl = is_rl
        self.rl_train_loader = rl_train_loader
        self._train_eval_loader = train_eval_loader
        if self.to.metrics is None:
            self.to.metrics = {}
        self._model = model
        self._val_loader = val_loader
        self._ctx = get_context(train_config.context, train_config.device_id)

        self._model = self._model.to(self._ctx)

        # define a summary writer that logs data and flushes to the file every 5 seconds
        if self.tc.log_metrics_to_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.sum_writer = SummaryWriter(log_dir=self.tc.export_dir+"logs", flush_secs=5)
        # Define the two loss functions
        if train_config.sparse_policy_label:
            self.policy_loss = SampleWeightedLoss(nn.CrossEntropyLoss)
        else:
            self.policy_loss = SampleWeightedLoss(SoftCrossEntropyLoss)
        self.value_loss = SampleWeightedLoss(nn.MSELoss)
        self.wdl_loss = SampleWeightedLoss(nn.CrossEntropyLoss)
        self.ply_loss = SampleWeightedLoss(nn.MSELoss)

        # Define the optimizer
        self.optimizer = create_optimizer(self._model, self.tc)

        self.ordering = glob.glob(main_config['planes_train_dir'] + '*.parquet')

        # For RL mode, we don't use the ordering (file-based loading)
        if self.is_rl and not self.ordering:
            self.ordering = ['rl_data']  # Placeholder to prevent empty ordering check

        # few variables which are internally used
        self.val_loss_best = self.val_p_acc_best = self.k_steps_best = \
            self.old_label = self.value_out = self.t_s = None
        self.patience_cnt = self.batch_proc_tmp = None
        # calculate how many log states will be processed
        if self.tc.full_eval_each_epoch:
            epoch_count = max(1, self.tc.nb_training_epochs)
            interval_capacity = epoch_count * self.tc.batch_steps
            evaluations_per_epoch = max(
                1,
                (self.tc.total_it + interval_capacity - 1) // interval_capacity,
            )
            self.k_steps_end = evaluations_per_epoch * epoch_count
        else:
            self.k_steps_end = round(self.tc.total_it / self.tc.batch_steps)
        if self.k_steps_end == 0:
            self.k_steps_end = 1
        self.k_steps = self.cur_it = self.nb_spikes = self.old_val_loss = self.continue_training = self.t_s_steps = None
        self._train_iter = self.graph_exported = self.val_metric_values = self.val_loss = self.val_p_acc = None
        self.val_metric_values_best = None
        self._generated_weight_files = []

        self.use_rtpt = use_rtpt

        if use_rtpt:
            # we use k-steps instead of epochs here
            self.rtpt = RTPT(name_initials=self.tc.name_initials, experiment_name='crazyara',
                             max_iterations=self.k_steps_end-self.tc.k_steps_initial)

    def train(self, cur_it=None):
        """
        Training model
        :param cur_it: Current iteration which is used for the learning rate and momentum schedule.
         If set to None it will be initialized
        :return: return_metrics_and_stop_training()
        """

        self._setup_variables(cur_it)
        set_train_mode(self._model, self.tc)  # set training mode

        while self.continue_training:
            # reshuffle the ordering of the training game batches (shuffle works in place)
            random.shuffle(self.ordering)

            self.epoch += 1
            logging.info("EPOCH %d", self.epoch)
            logging.info("=========================")
            self.t_s_steps = time()

            # RL mode: iterate over the rl_train_loader directly
            if self.is_rl and self.rl_train_loader is not None:
                for _, batch in enumerate(self.rl_train_loader):
                    data = self.train_update_rl(batch)

                    if not self.graph_exported and self.tc.log_metrics_to_tensorboard:
                        self.sum_writer.add_graph(self._model, data)
                        self.graph_exported = True

                    if self.batch_proc_tmp >= self.tc.batch_steps or self.cur_it >= self.tc.total_it:
                        train_metric_values, val_metric_values, additional_metric_values = self.evaluate_rl(self.rl_train_loader)
                        self._handle_evaluation(train_metric_values, val_metric_values, additional_metric_values, data)
                        if not self.continue_training or self.cur_it >= self.tc.total_it:
                            return self._finish_training(val_metric_values)
            else:
                # SL mode: iterate over shard files
                for shard_index, shard_file_path in enumerate(tqdm(self.ordering)):
                    train_loader = self._get_train_loader(shard_file_path)

                    for batch_index, batch in enumerate(train_loader):
                        data = self.train_update(batch)

                        # add the graph representation of the network to the tensorboard log file
                        if not self.graph_exported and self.tc.log_metrics_to_tensorboard:
                            self.sum_writer.add_graph(self._model, data)
                            self.graph_exported = True

                        epoch_complete = (
                            shard_index + 1 == len(self.ordering)
                            and batch_index + 1 == len(train_loader)
                        )
                        training_complete = self.cur_it >= self.tc.total_it
                        interval_complete = self.batch_proc_tmp >= self.tc.batch_steps
                        full_evaluation = requires_full_evaluation(
                            self.tc,
                            training_complete,
                            epoch_complete,
                        )
                        if interval_complete or full_evaluation:
                            train_metric_values, val_metric_values, additional_metric_values = self.evaluate(
                                train_loader,
                                full_evaluation=full_evaluation,
                            )
                        else:
                            continue

                        if self.use_rtpt:
                            # update process title according to loss
                            self.rtpt.step(subtitle=f"loss={val_metric_values['loss']:2.2f}")
                        if self.tc.use_spike_recovery and (
                                self.old_val_loss * self.tc.spike_thresh < val_metric_values["loss"]
                                or torch.isnan(val_metric_values["loss"])
                        ):  # check for spikes
                            self.nb_spikes += 1
                            logging.warning(
                                "Spike %d/%d occurred - val_loss: %.3f",
                                self.nb_spikes,
                                self.tc.max_spikes,
                                val_metric_values["loss"],
                            )
                            if self.nb_spikes >= self.tc.max_spikes:
                                self.val_loss = val_metric_values["loss"]
                                self.val_p_acc = val_metric_values["policy_acc"]
                                logging.debug("The maximum number of spikes has been reached. Stop training.")
                                # finally stop training because the number of lr drops has been achieved
                                print()
                                print(
                                    "Elapsed time for training(hh:mm:ss): "
                                    + str(datetime.timedelta(seconds=round(time() - self.t_s)))
                                )

                                if self.tc.log_metrics_to_tensorboard:
                                    self.sum_writer.close()
                                return return_metrics_and_stop_training(self.k_steps, val_metric_values, self.k_steps_best,
                                                                        self.val_metric_values_best)

                            logging.debug("Recover to latest checkpoint")
                            model_path = self.weights_dir / ("model-%.5f-%.3f-%04d.tar" % (
                                self.val_loss_best,
                                self.val_p_acc_best,
                                self.k_steps_best,
                            ))  # Load the best model once again
                            logging.debug("load current best model:%s", model_path)
                            load_torch_state(self._model, self.optimizer, model_path, self.tc.device_id)
                            self.k_steps = self.k_steps_best
                            logging.debug("k_step is back at %d", self.k_steps_best)
                            # print the elapsed time
                            self.t_delta = time() - self.t_s_steps
                            print(" - %.ds" % self.t_delta)
                            self.t_s_steps = time()
                        else:
                            # update the val_loss_value to compare with using spike recovery
                            self.old_val_loss = val_metric_values["loss"]
                            # log the metric values to tensorboard
                            self._log_metrics(train_metric_values, global_step=self.k_steps, prefix="train_")
                            self._log_metrics(val_metric_values, global_step=self.k_steps, prefix="val_")
                            if self.additional_loaders is not None:
                                for dataset_name, metric_values in additional_metric_values.items():
                                    self._log_metrics(metric_values, global_step=self.k_steps, prefix=f"{dataset_name}_")

                            if self.tc.log_metrics_to_tensorboard and self.tc.export_grad_histograms:
                                # logging the gradients of parameters for checking convergence
                                for name, param in self._model.named_parameters():
                                    if "bn" not in name and "batch" not in name and name != "policy_flat_plane_idx":
                                        self.sum_writer.add_histogram(
                                            tag=name, values=param, global_step=self.k_steps,
                                        )

                            # check if a new checkpoint shall be created
                            if self.val_loss_best is None or val_metric_values["loss"] < self.val_loss_best:
                                # update val_loss_best
                                self.val_loss_best = val_metric_values["loss"]
                                self.val_p_acc_best = val_metric_values["policy_acc"]
                                self.val_metric_values_best = val_metric_values
                                self.k_steps_best = self.k_steps

                                if self.tc.export_weights:
                                    model_prefix = "model-%.5f-%.3f-%04d"\
                                                   % (self.val_loss_best, self.val_p_acc_best, self.k_steps_best)
                                    filepath = self.weights_dir / f"{model_prefix}.tar"
                                    self.delete_previous_weights()

                                    # the export function saves both the architecture and the weights
                                    save_torch_state(
                                        self._model,
                                        self.optimizer,
                                        filepath,
                                        training_iteration=self.cur_it,
                                        evaluation_step=self.k_steps,
                                        batch_steps=self.tc.batch_steps,
                                    )
                                    print()
                                    logging.info("Saved checkpoint to %s", filepath)
                                    if self.tc.export_intermediate_onnx:
                                        with torch.no_grad():
                                            ctx = get_context(self.tc.context, self.tc.device_id)
                                            dummy_input = torch.zeros(
                                                1, data.shape[1], data.shape[2], data.shape[3]
                                            ).to(ctx)
                                            export_to_onnx(
                                                self._model, 1, dummy_input,
                                                self.weights_dir, model_prefix,
                                                self.tc.use_wdl and self.tc.use_plys_to_end,
                                                True,
                                            )
                                    self._generated_weight_files.extend(
                                        self.weights_dir.glob(f"{model_prefix}*")
                                    )

                                self.patience_cnt = 0  # reset the patience counter
                            # print the elapsed time
                            self.t_delta = time() - self.t_s_steps
                            print(" - %.ds" % self.t_delta)
                            self.t_s_steps = time()

                            if self.tc.log_metrics_to_tensorboard:
                                # log the samples per second metric to tensorboard
                                self.sum_writer.add_scalar(
                                    tag="samples_per_second",
                                    scalar_value=data.shape[0] * self.tc.batch_steps / self.t_delta,
                                    global_step=self.k_steps,
                                )

                                # log the current learning rate
                                self.sum_writer.add_scalar(tag="lr", scalar_value=self.to.lr_schedule(self.cur_it), global_step=self.k_steps)
                                # log the current momentum value
                                self.sum_writer.add_scalar(
                                    tag="momentum", scalar_value=self.to.momentum_schedule(self.cur_it), global_step=self.k_steps
                                )

                            if self.cur_it >= self.tc.total_it:
                                logging.debug("The number of given iterations has been reached")
                                # finally stop training because the number of lr drops has been achieved
                                print()
                                print(
                                    "Elapsed time for training(hh:mm:ss): "
                                    + str(datetime.timedelta(seconds=round(time() - self.t_s)))
                                )

                                if self.tc.log_metrics_to_tensorboard:
                                    self.sum_writer.close()

                                # make sure to empty cache
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                return return_metrics_and_stop_training(self.k_steps, val_metric_values, self.k_steps_best,
                                                                        self.val_metric_values_best)

    def delete_previous_weights(self):
        """Delete only checkpoints generated by this trainer instance."""
        for path in self._generated_weight_files:
            path.unlink(missing_ok=True)
        self._generated_weight_files.clear()

    def _get_train_loader(self, shard_file_path):
        tensors = load_parquet_shard(
            shard_file_path,
            include_auxiliary=self.tc.use_wdl and self.tc.use_plys_to_end,
        )

        # Create dataset and dataloader
        dataset = TensorDataset(*tensors)
        train_loader = DataLoader(dataset, batch_size=self.tc.batch_size, shuffle=True)

        return train_loader

    def evaluate(self, train_loader, full_evaluation=False):
        # log the current learning rate
        processed_batches = max(1, self.batch_proc_tmp)
        if full_evaluation:
            self.batch_proc_tmp = 0
        else:
            self.batch_proc_tmp -= self.tc.batch_steps
        ms_step = ((time() - self.t_s_steps) / processed_batches) * 1000  # measure elapsed time
        # update the counters
        self.k_steps += 1
        self.patience_cnt += 1
        logging.info("Step %dK/%dK - %dms/step", self.k_steps, self.k_steps_end, ms_step)
        logging.info("-------------------------")
        logging.debug("Iteration %d/%d", self.cur_it, self.tc.total_it)
        logging.debug("lr: %.7f - momentum: %.7f", self.to.lr_schedule(self.cur_it),
                      self.to.momentum_schedule(self.cur_it))
        eval_batch_limit = evaluation_batch_limit(self.tc, full_evaluation)
        evaluation_label = "full" if full_evaluation else "intermediate"
        print(f"starting {evaluation_label} train eval")
        evaluation_loader = (
            self._train_eval_loader
            if self._train_eval_loader is not None
            else train_loader
        )
        train_metric_values = evaluate_metrics(
            self.to.metrics,
            evaluation_loader,
            self._model,
            nb_batches=(
                eval_batch_limit
                if self._train_eval_loader is not None
                else 25
            ),
            ctx=self._ctx,
            phase_weights=self.to.phase_weights,
            sparse_policy_label=self.tc.sparse_policy_label,
            apply_select_policy_from_plane=self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data,
            use_wdl=self.tc.use_wdl,
            use_plys_to_end=self.tc.use_plys_to_end,
            precision=self.tc.mixed_precision,
        )

        print(f"starting {evaluation_label} val eval")
        val_metric_values = evaluate_metrics(
            self.to.metrics,
            self._val_loader,
            self._model,
            nb_batches=eval_batch_limit,
            ctx=self._ctx,
            phase_weights=self.to.phase_weights,
            sparse_policy_label=self.tc.sparse_policy_label,
            apply_select_policy_from_plane=self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data,
            use_wdl=self.tc.use_wdl,
            use_plys_to_end=self.tc.use_plys_to_end,
            precision=self.tc.mixed_precision,
        )

        # do additional evaluations based on self.additional_loaders
        additional_metric_values = dict()
        if self.additional_loaders is not None:
            for dataset_name, dataloader in self.additional_loaders.items():
                print(f"starting {dataset_name} eval")
                metric_values = evaluate_metrics(
                    self.to.metrics,
                    dataloader,
                    self._model,
                    nb_batches=eval_batch_limit,
                    ctx=self._ctx,
                    phase_weights={k: 1.0 for k, v in self.to.phase_weights.items()},  # use no weighting
                    sparse_policy_label=self.tc.sparse_policy_label,
                    apply_select_policy_from_plane=self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data,
                    use_wdl=self.tc.use_wdl,
                    use_plys_to_end=self.tc.use_plys_to_end,
                    precision=self.tc.mixed_precision,
                )
                additional_metric_values[dataset_name] = metric_values

        set_train_mode(self._model, self.tc)  # return back to training mode
        return train_metric_values, val_metric_values, additional_metric_values

    def train_update(self, batch):
        self.optimizer.zero_grad()

        if self.tc.use_wdl and self.tc.use_plys_to_end:
            data, value_label, policy_label, wdl_label, plys_label = batch
            wdl_label = wdl_label.to(self._ctx)
            plys_label = plys_label.to(self._ctx)
        else:
            data, value_label, policy_label = batch
        data = data.to(self._ctx)
        value_label = value_label.to(self._ctx)
        policy_label = policy_label.to(self._ctx)
        sample_weights = torch.ones(len(value_label), device=self._ctx)
        if self.tc.sparse_policy_label:
            policy_label = policy_label.long()
        # update a dummy metric to see a proper progress bar
        #  (the metrics will get evaluated at the end of 100k steps)
        # if self.batch_proc_tmp > 0:
        #     self.to.metrics["value_loss"].update(self.old_label, value_out)
        self.old_label = value_label

        with autocast_context(self._ctx, self.tc.mixed_precision):
            model_out = self._model(data)
        if self.tc.use_wdl and self.tc.use_plys_to_end:
            value_out, policy_out, _, wdl_out, plys_out = model_out[:5]
            wdl_loss = self.wdl_loss(wdl_out.float(), wdl_label, sample_weights)
            ply_loss = self.ply_loss(
                torch.flatten(plys_out.float()), plys_label, sample_weights
            )
        else:
            value_out, policy_out = model_out[:2]
        # policy_out = policy_out.softmax(dim=1)
        value_loss = self.value_loss(
            torch.flatten(value_out.float()), value_label, sample_weights
        )

        policy_out_1, policy_out_2 = policy_out
        policy_label_1 = policy_label[:, 0]
        policy_label_2 = policy_label[:, 1]

        policy_loss_1 = self.policy_loss(
            policy_out_1.float(), policy_label_1, sample_weights
        )
        policy_loss_2 = self.policy_loss(
            policy_out_2.float(), policy_label_2, sample_weights
        )

        policy_loss = 0.5 * (policy_loss_1 + policy_loss_2)

        # weight the components of the combined loss
        combined_loss = (
            self.tc.val_loss_factor * value_loss + self.tc.policy_loss_factor * policy_loss
        )
        if self.tc.use_wdl and self.tc.use_plys_to_end:
            combined_loss = (
                combined_loss
                + self.tc.wdl_loss_factor * wdl_loss
                + self.tc.plys_to_end_loss_factor * ply_loss
            )
        combined_loss.backward()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.to.lr_schedule(self.cur_it)  # update the learning rate
            if 'momentum' in param_group:
                param_group['momentum'] = self.to.momentum_schedule(self.cur_it)  # update the momentum
        self.optimizer.step()
        self.cur_it += 1
        self.batch_proc_tmp += 1
        return data

    def train_update_rl(self, batch):
        """
        Training update for RL mode with separate policy_a and policy_b targets.
        Batch format: (data, value_label, policy_a, policy_b, wdl, moves_left)
        """
        if getattr(self.tc, "freeze_batchnorm_stats", False):
            # Re-applied every step: tensorboard's add_graph (after the first
            # step) restores training mode with model.train(), which would put
            # every BatchNorm layer back into training mode.
            set_train_mode(self._model, self.tc)
        self.optimizer.zero_grad()

        data, value_label, policy_a, policy_b, wdl_label, moves_left_label = batch[:6]
        joint_targets, aux_labels, batch_weights = split_rl_batch_extras(
            batch, self.tc.aux_head_labels, uses_sample_weights(self.tc), with_weights=True
        )
        data = data.to(self._ctx)
        value_label = value_label.to(self._ctx)
        policy_a = policy_a.to(self._ctx)
        policy_b = policy_b.to(self._ctx)
        wdl_label = wdl_label.to(self._ctx)
        moves_left_label = moves_left_label.to(self._ctx)
        if joint_targets:
            joint_targets = tuple(target.to(self._ctx) for target in joint_targets)
        if aux_labels:
            aux_labels = tuple(label.to(self._ctx) for label in aux_labels)
        sample_weights = torch.ones(len(value_label), device=self._ctx)
        policy_weights = sample_weights
        if batch_weights is not None:
            policy_weights = batch_weights.to(self._ctx).float()
            if self.tc.sample_weight_targets == "all":
                sample_weights = policy_weights

        self.old_label = value_label

        with autocast_context(self._ctx, self.tc.mixed_precision):
            model_out = self._model(data)
            value_out, policy_out, _, wdl_out, moves_left_out = model_out[:5]
        value_loss = self.value_loss(
            torch.flatten(value_out.float()), value_label, sample_weights
        )

        policy_out_1, policy_out_2 = policy_out

        # For RL, we use soft cross-entropy with full policy distributions
        policy_loss_1 = self.policy_loss(
            policy_out_1.float(), policy_a, policy_weights
        )
        policy_loss_2 = self.policy_loss(
            policy_out_2.float(), policy_b, policy_weights
        )

        policy_loss = 0.5 * (policy_loss_1 + policy_loss_2)
        wdl_loss = self.wdl_loss(wdl_out.float(), wdl_label, sample_weights)
        moves_left_loss = self.ply_loss(
            torch.flatten(moves_left_out.float()), moves_left_label, sample_weights
        )
        joint_factors, aux_out = split_model_extras(self._model, model_out)
        joint_loss = value_loss.new_zeros(())
        if joint_targets and joint_factors is not None:
            joint_loss = joint_policy_cross_entropy(
                policy_out_1.float(),
                policy_out_2.float(),
                joint_factors[0].float(),
                joint_factors[1].float(),
                *joint_targets,
                rank=self.tc.joint_policy_rank,
                top_k=self.tc.joint_policy_top_k,
            )

        combined_loss = (
            self.tc.val_loss_factor * value_loss + self.tc.policy_loss_factor * policy_loss
            + self.tc.wdl_loss_factor * wdl_loss
            + self.tc.plys_to_end_loss_factor * moves_left_loss
            + self.tc.joint_policy_loss_factor * joint_loss
        )
        if aux_out is not None and aux_labels:
            think_loss, _, arrivals_loss, _ = aux_head_losses(
                *aux_out, *aux_labels, **arrivals_loss_options(self.tc)
            )
            combined_loss = (
                combined_loss
                + self.tc.think_gain_loss_factor * think_loss
                + self.tc.arrivals_loss_factor * arrivals_loss
            )
        combined_loss.backward()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.to.lr_schedule(self.cur_it) * param_group.get('lr_scale', 1.0)
            if 'momentum' in param_group:
                param_group['momentum'] = self.to.momentum_schedule(self.cur_it)
        self.optimizer.step()
        self.cur_it += 1
        self.batch_proc_tmp += 1
        return data

    def evaluate_rl(self, train_loader):
        """Evaluate metrics for RL training mode."""
        self.batch_proc_tmp -= self.tc.batch_steps
        ms_step = ((time() - self.t_s_steps) / self.tc.batch_steps) * 1000
        self.k_steps += 1
        self.patience_cnt += 1
        logging.info("Step %dK/%dK - %dms/step", self.k_steps, self.k_steps_end, ms_step)
        logging.info("-------------------------")
        logging.debug("Iteration %d/%d", self.cur_it, self.tc.total_it)
        logging.debug("lr: %.7f - momentum: %.7f", self.to.lr_schedule(self.cur_it),
                      self.to.momentum_schedule(self.cur_it))

        print("starting train eval (RL)")
        train_metric_values = self._evaluate_metrics_rl(train_loader, nb_batches=25)

        print("starting val eval (RL)")
        val_metric_values = self._evaluate_metrics_rl(self._val_loader, nb_batches=None)

        set_train_mode(self._model, self.tc)
        return train_metric_values, val_metric_values, {}

    def _evaluate_metrics_rl(self, data_loader, nb_batches=None):
        """
        Evaluate metrics for RL data with separate policy_a and policy_b.
        """
        reset_metrics(self.to.metrics)
        self._model.eval()
        joint_loss_sum = 0.0
        joint_loss_batches = 0
        aux_sums = [0.0, 0.0]  # think_gain BCE * labelled boards, arrivals MSE * labelled samples
        aux_counts = [0.0, 0.0]
        model_has_aux = getattr(self._model, "aux_heads", False)

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                data, value_label, policy_a, policy_b, wdl_label, moves_left_label = batch[:6]
                joint_targets, aux_labels = split_rl_batch_extras(
                    batch, self.tc.aux_head_labels, uses_sample_weights(self.tc)
                )
                data = data.to(self._ctx)
                value_label = value_label.to(self._ctx)
                policy_a = policy_a.to(self._ctx)
                policy_b = policy_b.to(self._ctx)
                wdl_label = wdl_label.to(self._ctx)
                moves_left_label = moves_left_label.to(self._ctx)
                if joint_targets:
                    joint_targets = tuple(
                        target.to(self._ctx) for target in joint_targets
                    )
                sample_weights = torch.ones(len(value_label), device=self._ctx)

                with autocast_context(self._ctx, self.tc.mixed_precision):
                    model_out = self._model(data)
                    value_out, policy_out, _, wdl_out, moves_left_out = model_out[:5]
                value_out = value_out.float()
                policy_out = tuple(output.float() for output in policy_out)
                wdl_out = wdl_out.float()
                moves_left_out = moves_left_out.float()
                joint_factors, aux_out = split_model_extras(self._model, model_out)
                if aux_out is not None and aux_labels:
                    think_loss, think_count, arrivals_loss, arrivals_count = aux_head_losses(
                        *aux_out, *(label.to(self._ctx) for label in aux_labels),
                        **arrivals_loss_options(self.tc)
                    )
                    aux_sums[0] += float(think_loss) * float(think_count)
                    aux_sums[1] += float(arrivals_loss) * float(arrivals_count)
                    aux_counts[0] += float(think_count)
                    aux_counts[1] += float(arrivals_count)
                if joint_targets and joint_factors is not None:
                    joint_loss_sum += float(joint_policy_cross_entropy(
                        policy_out[0], policy_out[1],
                        joint_factors[0].float(), joint_factors[1].float(),
                        *joint_targets,
                        rank=self.tc.joint_policy_rank,
                        top_k=self.tc.joint_policy_top_k,
                    ))
                    joint_loss_batches += 1

                # Update value metrics
                self.to.metrics["value_loss"].update(
                    preds=torch.flatten(value_out), labels=value_label, sample_weights=sample_weights
                )
                self.to.metrics["value_acc_sign"].update(
                    preds=torch.flatten(value_out), labels=value_label, sample_weights=sample_weights
                )
                self.to.metrics["wdl_loss"].update(
                    preds=wdl_out, labels=wdl_label, sample_weights=sample_weights
                )
                self.to.metrics["wdl_acc"].update(
                    preds=wdl_out.argmax(dim=1),
                    labels=wdl_label,
                    sample_weights=sample_weights,
                )
                self.to.metrics["plys_to_end_loss"].update(
                    preds=torch.flatten(moves_left_out),
                    labels=moves_left_label,
                    sample_weights=sample_weights,
                )

                # Update policy metrics
                # For soft labels, compare argmax of predictions with argmax of target distributions
                policy_out_1, policy_out_2 = policy_out
                self.to.metrics["policy_loss"].update(
                    preds=policy_out_1, labels=policy_a, sample_weights=sample_weights
                )
                # Accuracy metric with sparse_policy_label=False will call argmax on labels
                self.to.metrics["policy_acc"].update(
                    preds=policy_out_1.argmax(dim=1), labels=policy_a, sample_weights=sample_weights
                )
                self.to.metrics["policy_loss"].update(
                    preds=policy_out_2, labels=policy_b, sample_weights=sample_weights
                )
                self.to.metrics["policy_acc"].update(
                    preds=policy_out_2.argmax(dim=1), labels=policy_b, sample_weights=sample_weights
                )

                if nb_batches and i + 1 == nb_batches:
                    break

        joint_loss_value = torch.tensor(
            joint_loss_sum / max(1, joint_loss_batches), device=self._ctx
        )
        metric_values = {
            "loss": (
                self.tc.val_loss_factor * self.to.metrics["value_loss"].compute()
                + self.tc.policy_loss_factor * self.to.metrics["policy_loss"].compute()
                + self.tc.wdl_loss_factor * self.to.metrics["wdl_loss"].compute()
                + self.tc.plys_to_end_loss_factor
                    * self.to.metrics["plys_to_end_loss"].compute()
                + self.tc.joint_policy_loss_factor * joint_loss_value
            ),
        }
        # Reported only when the model has joint_factors heads: without them
        # the joint loss cannot be computed and a 0.0000 would mislead.
        if getattr(self._model, "joint_policy_heads", None) is not None:
            metric_values["joint_policy_loss"] = joint_loss_value
        if model_has_aux:
            think_value = torch.tensor(aux_sums[0] / max(1.0, aux_counts[0]), device=self._ctx)
            arrivals_value = torch.tensor(aux_sums[1] / max(1.0, aux_counts[1]), device=self._ctx)
            metric_values["loss"] = (
                metric_values["loss"]
                + self.tc.think_gain_loss_factor * think_value
                + self.tc.arrivals_loss_factor * arrivals_value
            )
            metric_values["think_gain_loss"] = think_value
            metric_values["think_gain_labels"] = torch.tensor(aux_counts[0], device=self._ctx)
            metric_values["arrivals_loss"] = arrivals_value
            metric_values["arrivals_labels"] = torch.tensor(aux_counts[1], device=self._ctx)
        for metric_name in self.to.metrics:
            metric_values[metric_name] = self.to.metrics[metric_name].compute()
        return metric_values

    def _handle_evaluation(self, train_metric_values, val_metric_values, additional_metric_values, data):
        """Handle post-evaluation logic (spike recovery, checkpointing, etc.)."""
        if self.use_rtpt:
            self.rtpt.step(subtitle=f"loss={val_metric_values['loss']:2.2f}")

        if self.tc.use_spike_recovery and (
                self.old_val_loss * self.tc.spike_thresh < val_metric_values["loss"]
                or torch.isnan(val_metric_values["loss"].detach())
        ):
            self.nb_spikes += 1
            logging.warning("Spike %d/%d occurred - val_loss: %.3f",
                           self.nb_spikes, self.tc.max_spikes, val_metric_values["loss"])
            if self.nb_spikes >= self.tc.max_spikes:
                self.val_loss = val_metric_values["loss"]
                self.val_p_acc = val_metric_values["policy_acc"]
                logging.debug("Maximum spikes reached. Stopping training.")
                self.continue_training = False
                return
            # Recovery logic
            if self.val_loss_best is not None:
                model_path = self.weights_dir / ("model-%.5f-%.3f-%04d.tar" % (
                    self.val_loss_best, self.val_p_acc_best, self.k_steps_best))
                logging.debug("Recovering from checkpoint: %s", model_path)
                load_torch_state(self._model, self.optimizer, model_path, self.tc.device_id)
                self.k_steps = self.k_steps_best
        else:
            self.old_val_loss = val_metric_values["loss"]
            self._log_metrics(train_metric_values, global_step=self.k_steps, prefix="train_")
            self._log_metrics(val_metric_values, global_step=self.k_steps, prefix="val_")

            if self.val_loss_best is None or val_metric_values["loss"] < self.val_loss_best:
                self.val_loss_best = val_metric_values["loss"]
                self.val_p_acc_best = val_metric_values["policy_acc"]
                self.val_metric_values_best = val_metric_values
                self.k_steps_best = self.k_steps

                if self.tc.export_weights:
                    model_prefix = "model-%.5f-%.3f-%04d" % (
                        self.val_loss_best, self.val_p_acc_best, self.k_steps_best)
                    filepath = self.weights_dir / f"{model_prefix}.tar"
                    self.delete_previous_weights()
                    save_torch_state(
                        self._model,
                        self.optimizer,
                        filepath,
                        training_iteration=self.cur_it,
                        evaluation_step=self.k_steps,
                        batch_steps=self.tc.batch_steps,
                    )
                    print()
                    logging.info("Saved checkpoint to %s", filepath)
                    if self.tc.export_intermediate_onnx:
                        with torch.no_grad():
                            ctx = get_context(self.tc.context, self.tc.device_id)
                            dummy_input = torch.zeros(
                                1, data.shape[1], data.shape[2], data.shape[3]
                            ).to(ctx)
                            export_to_onnx(
                                self._model, 1, dummy_input,
                                self.weights_dir, model_prefix,
                                self.tc.use_wdl and self.tc.use_plys_to_end, True,
                            )
                    self._generated_weight_files.extend(self.weights_dir.glob(f"{model_prefix}*"))

                self.patience_cnt = 0

            self.t_delta = time() - self.t_s_steps
            print(" - %.ds" % self.t_delta)
            self.t_s_steps = time()

    def _finish_training(self, val_metric_values):
        """Clean up and return final metrics."""
        print()
        print("Elapsed time for training(hh:mm:ss): " +
              str(datetime.timedelta(seconds=round(time() - self.t_s))))
        if self.tc.log_metrics_to_tensorboard:
            self.sum_writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return return_metrics_and_stop_training(
            self.k_steps, val_metric_values, self.k_steps_best, self.val_metric_values_best)

    def _log_metrics(self, metric_values, global_step, prefix="train_"):
        """
        Logs a dictionary object of metric value to the console and to tensorboard
        if _log_metrics_to_tensorboard is set to true
        :param metric_values: Dictionary object storing the current metrics
        :param global_step: X-Position point of all metric entries
        :param prefix: Used for labelling the metrics
        :return:
        """
        for name in metric_values.keys():  # show the metric stats
            print(" - %s%s: %.4f" % (prefix, name, metric_values[name]), end="")
            # add the metrics to the tensorboard event file
            if self.tc.log_metrics_to_tensorboard:
                self.sum_writer.add_scalar(tag="lr", scalar_value=self.to.lr_schedule(self.cur_it),
                                           global_step=self.k_steps)

                self.sum_writer.add_scalar(tag=f'{name}/{prefix.replace("_", "")}', scalar_value=metric_values[name], global_step=global_step)

    def _setup_variables(self, cur_it):
        if self.tc.seed is not None:
            random.seed(self.tc.seed)
        # define and initialize the variables which will be used
        self.t_s = time()
        # track on how many batches have been processed in this epoch
        self.patience_cnt = self.epoch = self.batch_proc_tmp = 0
        self.k_steps = self.tc.k_steps_initial  # counter for thousands steps
        if cur_it is None:
            self.cur_it = self.tc.k_steps_initial * 1000
        else:
            self.cur_it = cur_it
        self.nb_spikes = 0  # count the number of spikes that have been detected
        # initialize the loss to compare with, with a very high value
        self.old_val_loss = 9000
        self.graph_exported = False  # create a state variable to check if the net architecture has been reported yet
        self.continue_training = True
        self.optimizer.lr = self.to.lr_schedule(self.cur_it)
        if self.tc.optimizer_name == "nag":
            self.optimizer.momentum = self.to.momentum_schedule(self.cur_it)
        if not self.ordering:  # safety check to prevent eternal loop
            raise Exception("You must have at least one part file in your planes-dataset directory!")
        if self.use_rtpt:
            # Start the RTPT tracking
            self.rtpt.start()


def create_optimizer(model: nn.Module, train_config: TrainConfig):
    if getattr(train_config, "finetune_param_groups", False):
        if train_config.optimizer_name not in ("nag", "sgd"):
            raise ValueError(
                "finetune_param_groups scales gradients per input column, which matches a per-column "
                f"learning rate only for SGD/NAG, not {train_config.optimizer_name}"
            )
        groups = finetune_param_groups(model, train_config)
        return torch.optim.SGD(groups, lr=train_config.max_lr, momentum=train_config.max_momentum,
                               weight_decay=train_config.wd)
    if train_config.optimizer_name == "nag":  # torch.optim.SGD uses Nestorov momentum already
        return torch.optim.SGD(model.parameters(), lr=train_config.max_lr, momentum=train_config.max_momentum,
                               weight_decay=train_config.wd)
    elif train_config.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=train_config.max_lr, weight_decay=train_config.wd)
    elif train_config.optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=train_config.max_lr, weight_decay=train_config.wd)
    raise Exception(f"Selected optimizer {train_config.optimizer_name} is not supported.")


AUX_HEAD_PREFIXES = ("think_gain_head.", "arrivals_head.")
STEM_KEYS = ("body_spatial.0.body.0.weight", "shared_stem.body.0.weight")


def _is_board_token_projection(name: str) -> bool:
    return name.startswith("board_token_projections.") and name.endswith(".0.weight")


def new_input_columns(model: nn.Module, source_board_channels: int) -> dict:
    """{parameter name: bool mask over dim 1} for every tensor that reads the
    raw planes, True on the columns added by widening (board channels
    source_board_channels.. of each board block; tools/make_clock_net.py).
    Empty for a network that was not widened."""
    total = None
    for name, param in model.named_parameters():
        if name == "body_spatial.0.body.0.weight":
            total = int(param.shape[1]) // 2
        elif name == "shared_stem.body.0.weight":
            total = int(param.shape[1])
    board = getattr(model, "board_channels", None) or total
    if board is None or board <= source_board_channels:
        return {}
    masks = {}
    for name, param in model.named_parameters():
        width = int(param.shape[1]) if param.dim() > 1 else 0
        mask = None
        if name == "body_spatial.0.body.0.weight":
            mask = torch.zeros(width, dtype=torch.bool)
            mask[source_board_channels:board] = True
            mask[board + source_board_channels:2 * board] = True
        elif name == "shared_stem.body.0.weight":
            mask = torch.zeros(width, dtype=torch.bool)
            mask[source_board_channels:board] = True
        elif _is_board_token_projection(name):
            trunk = width - board
            mask = torch.zeros(width, dtype=torch.bool)
            mask[trunk + source_board_channels:trunk + board] = True
        if mask is not None:
            masks[name] = mask
    return masks


def finetune_param_groups(model: nn.Module, train_config: TrainConfig) -> list:
    """SGD parameter groups for the spec v3 fine-tuning recipe.

    trunk: every parameter the source network had, lr_scale = trunk_max_lr /
    max_lr (at most 1). new: the aux heads, lr_scale = new_params_lr_factor.
    widened: the input tensors with new columns, lr_scale =
    new_columns_lr_factor for the new columns; a gradient
    hook scales their old columns to the trunk rate and applies weight decay
    per column (the group's own weight_decay is 0), so for SGD/NAG every
    column moves exactly as if it sat in its own group. Each group carries
    'name' and 'param_names'; the trainer sets lr = schedule * lr_scale.
    """
    max_lr = float(train_config.max_lr)
    trunk_scale = min(1.0, float(train_config.trunk_max_lr) / max_lr) if max_lr > 0 else 1.0
    new_scale = float(train_config.new_params_lr_factor)
    column_scale_lr = float(getattr(train_config, "new_columns_lr_factor", new_scale))
    wd = float(train_config.wd)
    masks = new_input_columns(model, int(train_config.source_board_channels))
    groups = {
        "trunk": {"params": [], "param_names": [], "lr_scale": trunk_scale, "weight_decay": wd},
        "new": {"params": [], "param_names": [], "lr_scale": new_scale, "weight_decay": wd},
        "widened": {"params": [], "param_names": [], "lr_scale": column_scale_lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(AUX_HEAD_PREFIXES):
            key = "new"
        elif name in masks:
            key = "widened"
            shape = [1] * param.dim()
            shape[1] = -1
            column_scale = torch.where(
                masks[name], torch.tensor(1.0),
                torch.tensor(trunk_scale / column_scale_lr if column_scale_lr else 0.0),
            ).view(shape).to(param.device, param.dtype)

            def hook(grad, p=param, scale=column_scale):
                return (grad + wd * p.detach()) * scale

            param.register_hook(hook)
        else:
            key = "trunk"
        groups[key]["params"].append(param)
        groups[key]["param_names"].append(name)
    out = []
    for key, group in groups.items():
        if group["params"]:
            group["name"] = key
            out.append(group)
    return out


def set_train_mode(model: nn.Module, train_config: TrainConfig) -> None:
    """model.train(), except that with freeze_batchnorm_stats every BatchNorm
    layer stays in eval mode: it normalises with the source network's running
    statistics and does not update them (their affine weights still train)."""
    model.train()
    if getattr(train_config, "freeze_batchnorm_stats", False):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def uses_sample_weights(train_config: TrainConfig) -> bool:
    return getattr(train_config, "sample_weight_targets", "none") in ("policy", "all")


def arrivals_loss_options(train_config: TrainConfig) -> dict:
    return {
        "arrivals_loss": getattr(train_config, "arrivals_loss", "mse"),
        "piece_weights": getattr(train_config, "arrivals_piece_weights", None),
    }


class SoftCrossEntropyLoss(_Loss):
    """
    Computes cross entropy loss for continuous target distribution.
    # https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/18
    # or use BCELoss
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(SoftCrossEntropyLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_softmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(-target * log_softmax(input), 1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return torch.mean(loss)


AUX_LABEL_FIELDS = 4  # think_gain [2], think_gain_mask [2], arrivals [20], arrivals_mask []


def split_rl_batch_extras(batch, aux_head_labels: bool, sample_weights: bool = False,
                          with_weights: bool = False):
    """Split an RL batch after its six base fields.

    Layout: (data, value, policy_a, policy_b, wdl, moves_left), then the four
    joint-policy targets when the loader includes them, then, when
    aux_head_labels is set, (think_gain, think_gain_mask, arrivals,
    arrivals_mask), then, when sample_weights is set, the per-sample weight.
    Returns (joint_targets or None, aux_labels or None), plus the weights (or
    None) with with_weights.
    """
    extras = tuple(batch[6:])
    weights = None
    if sample_weights:
        if not extras:
            raise ValueError("RL batch has no sample-weight field")
        weights = extras[-1]
        extras = extras[:-1]
    aux_labels = None
    if aux_head_labels:
        if len(extras) < AUX_LABEL_FIELDS:
            raise ValueError("RL batch has no aux-head label fields")
        aux_labels = extras[-AUX_LABEL_FIELDS:]
        extras = extras[:-AUX_LABEL_FIELDS]
    if with_weights:
        return (extras or None), aux_labels, weights
    return (extras or None), aux_labels


def split_model_extras(model, model_out):
    """(joint factors or None, (think_gain, arrivals) or None) from a model's outputs.

    Output order: value, policy, auxiliary, wdl, moves_left, then
    joint_factors_a/b when the model has joint heads, then think_gain and
    arrivals when it has the spec v2 aux heads.
    """
    index = 5
    joint = None
    if getattr(model, "joint_policy_heads", None) is not None:
        joint = tuple(model_out[index:index + 2])
        index += 2
    aux = None
    if getattr(model, "aux_heads", False):
        aux = tuple(model_out[index:index + 2])
    return joint, aux


def aux_head_losses(think_logits: Tensor, arrivals_pred: Tensor,
                    think_label: Tensor, think_mask: Tensor,
                    arrivals_label: Tensor, arrivals_mask: Tensor,
                    arrivals_loss: str = "mse", piece_weights=None):
    """Masked losses of the spec v2 aux heads.

    think_gain: binary cross-entropy on the logits, averaged over the boards
    that carry a label (think_mask 1). arrivals: a per-value loss averaged
    over the 20 values (weighted by piece_weights, one per piece type P N B R
    Q, repeated for the four pockets, normalised to mean 1), then over the
    samples that carry a label (arrivals_mask 1). arrivals_loss: "mse" on the
    raw counts (spec v2), "log1p" (squared error to log1p(count); the output is
    log1p of the expected count) or "poisson" (Poisson NLL with the output as
    the log rate, without the constant log(k!) term).
    With no labels a loss is exactly 0 and passes no gradient.
    Returns (think_loss, labelled boards, arrivals_loss, labelled samples).
    """
    think_mask = think_mask.float()
    think_terms = nn.functional.binary_cross_entropy_with_logits(
        think_logits.float(), think_label.float(), reduction="none"
    ) * think_mask
    think_count = think_mask.sum()
    think_loss = think_terms.sum() / think_count.clamp_min(1.0)

    arrivals_mask = arrivals_mask.float().reshape(-1)
    pred = arrivals_pred.float()
    label = arrivals_label.float()
    if arrivals_loss == "mse":
        per_value = (pred - label) ** 2
    elif arrivals_loss == "log1p":
        per_value = (pred - torch.log1p(label.clamp_min(0.0))) ** 2
    elif arrivals_loss == "poisson":
        per_value = torch.exp(pred) - label * pred
    else:
        raise ValueError(f"Unknown arrivals_loss {arrivals_loss!r}")
    if piece_weights is not None:
        weights = torch.as_tensor(piece_weights, dtype=per_value.dtype, device=per_value.device)
        weights = weights.repeat(per_value.shape[1] // weights.numel())
        per_value = per_value * (weights / weights.mean())
    arrivals_terms = per_value.mean(dim=1) * arrivals_mask
    arrivals_count = arrivals_mask.sum()
    arrivals_loss = arrivals_terms.sum() / arrivals_count.clamp_min(1.0)
    return think_loss, think_count, arrivals_loss, arrivals_count


def joint_policy_cross_entropy(
    policy_a: Tensor,
    policy_b: Tensor,
    factors_a: Tensor,
    factors_b: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    target_probability: Tensor,
    target_count: Tensor,
    *,
    rank: int,
    top_k: int,
) -> Tensor:
    """Residual joint-policy loss with marginal Cartesian hard negatives."""
    if rank <= 0:
        return factors_a.sum() * 0.0

    batch_size = policy_a.shape[0]
    vocabulary_size = factors_a.shape[1] // rank
    factors_a = factors_a.reshape(batch_size, rank, vocabulary_size).transpose(1, 2)
    factors_b = factors_b.reshape(batch_size, rank, vocabulary_size).transpose(1, 2)

    target_a = target_a.long()
    target_b = target_b.long()
    target_probability = target_probability.float()
    positions = torch.arange(target_a.shape[1], device=target_a.device)
    target_valid = positions.unsqueeze(0) < target_count.long().unsqueeze(1)

    selected = min(top_k, vocabulary_size)
    masked_target_probability = target_probability * target_valid
    target_marginal_a = target_probability.new_zeros(
        batch_size, vocabulary_size
    ).scatter_add(1, target_a, masked_target_probability)
    target_marginal_b = target_probability.new_zeros(
        batch_size, vocabulary_size
    ).scatter_add(1, target_b, masked_target_probability)
    marginal_probability_a, marginal_a = torch.topk(
        target_marginal_a, selected, dim=1
    )
    marginal_probability_b, marginal_b = torch.topk(
        target_marginal_b, selected, dim=1
    )
    cross_a = marginal_a.unsqueeze(2).expand(-1, -1, selected).reshape(batch_size, -1)
    cross_b = marginal_b.unsqueeze(1).expand(-1, selected, -1).reshape(batch_size, -1)
    cross_valid = (
        marginal_probability_a.unsqueeze(2)
        * marginal_probability_b.unsqueeze(1)
    ).reshape(batch_size, -1) > 0
    duplicate = (
        (cross_a.unsqueeze(2) == target_a.unsqueeze(1))
        & (cross_b.unsqueeze(2) == target_b.unsqueeze(1))
        & target_valid.unsqueeze(1)
    ).any(dim=2)

    candidate_a = torch.cat((target_a, cross_a), dim=1)
    candidate_b = torch.cat((target_b, cross_b), dim=1)
    candidate_valid = torch.cat((target_valid, cross_valid & ~duplicate), dim=1)

    def gather_factors(factors, indices):
        return torch.gather(
            factors,
            1,
            indices.unsqueeze(2).expand(-1, -1, rank),
        )

    compatibility = (
        gather_factors(factors_a, candidate_a)
        * gather_factors(factors_b, candidate_b)
    ).sum(dim=2) / (rank ** 0.5)
    marginal_index_a = candidate_a.masked_fill(
        candidate_a == vocabulary_size - 1, 0
    )
    marginal_index_b = candidate_b.masked_fill(
        candidate_b == vocabulary_size - 1, 0
    )
    logits = (
        torch.gather(policy_a, 1, marginal_index_a)
        + torch.gather(policy_b, 1, marginal_index_b)
        + compatibility
    ).masked_fill(~candidate_valid, -1.0e9)
    target_log_probability = torch.log_softmax(logits, dim=1)[
        :, :target_a.shape[1]
    ]
    per_sample = -(
        target_probability * target_log_probability * target_valid
    ).sum(dim=1)
    available = target_count > 0
    if not bool(available.any()):
        return factors_a.sum() * 0.0
    return per_sample[available].mean()


class SampleWeightedLoss(nn.Module):
    def __init__(self, loss_module):
        super(SampleWeightedLoss, self).__init__()
        self.criterion = loss_module(reduction="none")

    def forward(self, input: Tensor, target: Tensor, sample_weights: Tensor) -> Tensor:
        losses = self.criterion(input, target)
        weighted_losses = losses * sample_weights
        return torch.mean(weighted_losses)


def get_context(context: str, device_id: int):
    """
    Returns the computation context as  Pytorch device object.
    :param context: Computational context either "gpu" or "cpu"
    :param device_id: Device index to use (only relevant for context=="gpu")
    :return: Pytorch device object
    """
    if context == "gpu":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_id}")
        logging.info("No cuda device available. Fallback to CPU")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def load_torch_state(model: nn.Module, optimizer: Optimizer, path: Path, device_id: int):
    checkpoint = torch.load(path, map_location=f"cuda:{device_id}")
    restore_torch_state(model, optimizer, checkpoint)
    return checkpoint


def restore_torch_state(model: nn.Module, optimizer: Optimizer, checkpoint):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def save_torch_state(
    model: nn.Module,
    optimizer: Optimizer,
    path: Path,
    *,
    training_iteration: int = None,
    evaluation_step: int = None,
    batch_steps: int = None,
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if training_iteration is not None:
        checkpoint['training_iteration'] = training_iteration
    if evaluation_step is not None:
        checkpoint['evaluation_step'] = evaluation_step
    if batch_steps is not None:
        checkpoint['batch_steps'] = batch_steps
    torch.save(checkpoint, path)


def export_model(model, batch_sizes, input_shape, dir=Path('.'), torch_cpu=True, torch_cuda=True, onnx=True,
                 verbose=False):
    """
    Exports the model in ONNX and Torch Script Module.

    :param model: Pytorch model
    :param batch_sizes: List of batch sizes to use for export
    :param input_shape: Input shape of the model
    :param dir: The base path for all models
    :param torch_cpu: Whether to export as script module with cpu inputs
    :param torch_cuda: Whether to export as script module with cuda inputs
    :param onnx: Whether to export as onnx
    :param verbose: Print debug information
    """

    if dir.exists():
        # make sure that all the content is deleted first so we don't run into strange caching issues
        dir.rm_dir(dir, keep_empty_dir=False)

    dir.mkdir(parents=True, exist_ok=False)

    onnx_dir = dir / "onnx"
    if torch_cpu:
        onnx_dir.mkdir(parents=True, exist_ok=False)

    cpu_dir = dir / "torch_cpu"
    if torch_cpu:
        cpu_dir.mkdir(parents=True, exist_ok=False)

    torch_cuda = torch_cuda and torch.cuda.is_available()
    cuda_dir = dir / "torch_cuda"
    if torch_cuda:
        cuda_dir.mkdir(parents=True, exist_ok=False)

    model = model.eval()

    for batch_size in batch_sizes:
        dummy_input = torch.ones(batch_size, input_shape[0], input_shape[1], input_shape[2], dtype=torch.float)

        if model.is_stateful:
            dummy_input = model.flatten(dummy_input, model.get_init_state_bf_flat(batch_size, "cpu"))
            model.set_input_options(sequence_length=None, has_state_input=True)
        else:
            dummy_input = model.flatten(dummy_input, None)
            model.set_input_options(sequence_length=None, has_state_input=False)

        if onnx:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_to_onnx(model, batch_size, dummy_input, onnx_dir)

        if torch_cpu:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_as_script_module(model, batch_size, dummy_input, cpu_dir)

        if torch_cuda:
            dummy_input = dummy_input.cuda()
            model = model.cuda()
            export_as_script_module(model, batch_size, dummy_input, cuda_dir)

        if verbose:
            print("Input shape: ")
            print(dummy_input.shape)
            print("Output shape: ")
            for i, e in enumerate(model(dummy_input)):
                print(f"{i}: {e.shape}")


def export_to_onnx(model, batch_size: int, dummy_input: torch.Tensor, dir: Path, model_prefix: str,
                   has_auxiliary_output: bool, dynamic_batch_size: bool, input_version=None,
                   fp16: bool = True, onnx_name: str = None) -> Path:
    """
    Exports the model to ONNX format to allow later import in TensorRT.

    :param model: Pytorch model
    :param batch_size: The batch size of the input
    :param dummy_input: Dummy input which defines the input shape for the model
    :param dir: Output directory
    :param model_prefix: Model prefix name
    :param has_auxiliary_output: Determines if the model has an auxiliary output
    :param dynamic_batch_size: Whether to export model with dynamic batch size
    :param input_version: Can be used to specify the input representation version e.g. "3.0" for chess models.
    If none, the version will be read from the main_config file instead. It is used for labelling the onnx file.
    :param fp16: Convert the simplified graph to FP16 with FP32 inputs/outputs (default, as before). False keeps FP32.
    :param onnx_name: Exact file name to write instead of the generated one.
    :return: Path of the written file
    """
    if has_auxiliary_output:
        input_names = ["data"]
        output_names = [
            "value",
            "pi_a",
            "pi_b",
            main_config["auxiliary_output"],
            main_config["wdl_output"],
            "moves_left",
        ]
    else:
        input_names = ["obs"]
        output_names = ["value", "pi_a", "pi_b"]

    if getattr(model, "joint_policy_rank", 0) > 0:
        output_names.extend(("joint_factors_a", "joint_factors_b"))
    if getattr(model, "aux_heads", False):
        # Spec v2: think_gain = one logit per board (A, B); arrivals = 20
        # predicted counts (ours A, theirs A, ours B, theirs B) x (P N B R Q).
        output_names.extend(("think_gain", "arrivals"))

    dynamic_axes = None
    if dynamic_batch_size:
        dynamic_axes = {
            name: {0: "batch"}
            for name in [*input_names, *output_names]
        }

    if input_version is None:
        input_version = f"{main_config['version']}.0"

    if onnx_name is None:
        onnx_name = f"{model_prefix}-v{input_version}"
        if not dynamic_batch_size:
            onnx_name += f"-bsize-{batch_size}"
        onnx_name += ".onnx"

    model_filepath = str(dir / Path(onnx_name))

    model.eval()
    if hasattr(model, 'merge_bn'):
        model.merge_bn()

    torch.onnx.export(model, (dummy_input,), model_filepath,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      dynamo=False,
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=False)

    # simplify ONNX model
    # https://github.com/daquexian/onnx-simplifier
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(model_filepath)
    model_simp, check = simplify(model)
    if not check:
        raise Exception("Simplified ONNX model could not be validated")
    model_fp16 = (
        convert_float_to_float16(model_simp, keep_io_types=True)
        if fp16
        else model_simp
    )
    topologically_sort_graph(model_fp16)
    onnx.checker.check_model(model_fp16)
    if has_auxiliary_output:
        outputs = [output for output in model_fp16.graph.output
                   if output.name != main_config["auxiliary_output"]]
        del model_fp16.graph.output[:]
        model_fp16.graph.output.extend(outputs)
    onnx.save(model_fp16, model_filepath)
    return Path(model_filepath)


def export_as_script_module(model, batch_size, dummy_input, dir) -> None:
    """
    Exports the model to a Torch Script Module to allow later import in C++.

    :param model: Pytorch model
    :param batch_size: The batch size of the input
    :param dummy_input: Dummy input which defines the input shape for the model
    :return:
    """

    # generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, dummy_input)

    # serialize script module to file
    traced_script_module.save(str(dir / Path(f"model-bsize-{batch_size}.pt")))


def reset_metrics(metrics):
    """
    Resets all metric entries in a dictionary object
    :param metrics:
    :return:
    """
    for metric in metrics.values():
        metric.reset()


def evaluate_metrics(metrics, data_iterator, model, nb_batches, ctx, phase_weights, sparse_policy_label=False,
                     apply_select_policy_from_plane=True, use_wdl=False, use_plys_to_end=False,
                     precision="fp32"):
    """
    Runs inference of the network on a data_iterator object and evaluates the given metrics.
    The metric results are returned as a dictionary object.

    :param metrics: List of mxnet metrics which must have the
    names ['value_loss', 'policy_loss', 'value_acc_sign', 'policy_acc']
    :param data_iterator: Pytorch data iterator object
    :param model: Pytorch model handle
    :param nb_batches: Number of batches to evaluate (early stopping).
     If set to None all batches of the data_iterator will be evaluated
    :param ctx: Pytorch data context
    :param phase_weights: dictionary with the weights of each phase as phase: weight kv pairs
    :param sparse_policy_label: Should be set to true if the policy uses one-hot encoded targets
     (e.g. supervised learning)
    :param apply_select_policy_from_plane: If true, given policy label is converted to policy map index
    :return: Metric values
    """
    reset_metrics(metrics)
    model.eval()  # set model to evaluation mode
    with torch.no_grad():  # operations inside don't track history
        print("eval iterator length:", len(data_iterator), "eval phase weights:", phase_weights)
        for i, batch in enumerate(data_iterator):
            if use_wdl and use_plys_to_end:
                data, value_label, policy_label, wdl_label, plys_label = batch
                plys_label = plys_label.to(ctx)
                wdl_label = wdl_label.to(ctx).long()
            else:
                data, value_label, policy_label  = batch
            data = data.to(ctx)
            value_label = value_label.to(ctx)
            policy_label = policy_label.to(ctx)
            sample_weights = torch.ones(len(value_label), device=ctx)

            with autocast_context(ctx, precision):
                model_out = model(data)
            if use_wdl and use_plys_to_end:
                value_out, policy_out, _, wdl_out, plys_out = model_out[:5]
                wdl_out = wdl_out.float()
                plys_out = plys_out.float()
                metrics["wdl_loss"].update(preds=wdl_out, labels=wdl_label, sample_weights=sample_weights)
                metrics["wdl_acc"].update(preds=wdl_out.argmax(axis=1), labels=wdl_label, sample_weights=sample_weights)
                metrics["plys_to_end_loss"].update(preds=torch.flatten(plys_out), labels=plys_label,
                                                   sample_weights=sample_weights)
            else:
                value_out, policy_out = model_out

            value_out = value_out.float()
            policy_out = tuple(output.float() for output in policy_out)

            # update the metrics
            metrics["value_loss"].update(preds=torch.flatten(value_out), labels=value_label,
                                         sample_weights=sample_weights)
            metrics["value_acc_sign"].update(preds=torch.flatten(value_out), labels=value_label,
                                             sample_weights=sample_weights)

            metrics["policy_loss"].update(preds=policy_out[0],  # .softmax(dim=1),
                                          labels=policy_label[:, 0], sample_weights=sample_weights)
            metrics["policy_acc"].update(preds=policy_out[0].argmax(axis=1),
                                         labels=policy_label[:, 0], sample_weights=sample_weights)

            metrics["policy_loss"].update(preds=policy_out[1],  # .softmax(dim=1),
                                          labels=policy_label[:, 1], sample_weights=sample_weights)
            metrics["policy_acc"].update(preds=policy_out[1].argmax(axis=1),
                                         labels=policy_label[:, 1], sample_weights=sample_weights)

            # stop after evaluating x batches (only recommended to use this for the train set evaluation)
            if nb_batches and i+1 == nb_batches:
                break

    value_weight = 0.01
    policy_weight = 0.978 if use_wdl and use_plys_to_end else 0.99
    total_loss = (
        value_weight * metrics["value_loss"].compute()
        + policy_weight * metrics["policy_loss"].compute()
    )
    if use_wdl and use_plys_to_end:
        total_loss = (
            total_loss
            + 0.01 * metrics["wdl_loss"].compute()
            + 0.002 * metrics["plys_to_end_loss"].compute()
        )
    metric_values = {"loss": total_loss}

    for metric_name in metrics:
        metric_values[metric_name] = metrics[metric_name].compute()
    return metric_values


def get_data_loader(pgn_dataset_arrays_dict: dict, tc: TrainConfig, shuffle=True):
    """
    Returns a DataLoader object for the given numpy arrays.
    !Note: This function modifies the y_policy!
    :param pgn_dataset_arrays_dict: Dict object containing the numpy arrays of load_pgn_dataset
    :param tc: Training config object
    :param shuffle: Decide whether to shuffle the dataset or not
    :return: Returns the data loader object
    """
    d = pgn_dataset_arrays_dict


    dataset = TensorDataset(torch.Tensor(d['x']), torch.Tensor(d['y_value']), torch.Tensor(d['y_policy']))
    train_loader = DataLoader(dataset, shuffle=shuffle, batch_size=tc.batch_size, num_workers=tc.cpu_count)
    return train_loader


# ------------------------------------------------------------------ drift (spec v3 WP-D)
# How far training moved the network from it04 on positions it04 knows: the
# frozen supervised validation shard (or any shard of planes), with every plane
# past the source's 37 per board set to 0 (no clocks, as the engine sees a
# position without clocks), so the widened start network equals it04 exactly.
DRIFT_POLICY_KL_MAX = 0.05
DRIFT_VALUE_MSE_MAX = 0.01


def narrow_board_planes(x: Tensor, board_channels: int) -> Tensor:
    """Keep the first board_channels channels of each of the two board blocks."""
    per_board = x.shape[1] // 2
    if per_board == board_channels:
        return x
    return torch.cat((x[:, :board_channels], x[:, per_board:per_board + board_channels]), dim=1)


def widen_board_planes(x: Tensor, board_channels: int) -> Tensor:
    """Pad each board block with zero planes up to board_channels."""
    per_board = x.shape[1] // 2
    if per_board == board_channels:
        return x
    out = x.new_zeros((x.shape[0], 2 * board_channels, *x.shape[2:]))
    out[:, :per_board] = x[:, :per_board]
    out[:, board_channels:board_channels + per_board] = x[:, per_board:]
    return out


def load_drift_inputs(path, board_channels: int, max_samples: int = 1024,
                      source_board_channels: int = 37) -> Tensor:
    """Normalised planes (N, 2 * board_channels, 8, 8) from the 'x' column of a
    supervised or RL Parquet shard, first max_samples rows, every channel from
    source_board_channels on (per board) set to 0."""
    import numpy as np
    import polars as pl
    from hivemind.training.data_loaders import infer_input_channels

    frame = pl.read_parquet(path, columns=["x"], n_rows=max_samples)
    values = frame["x"].to_list()
    channels = infer_input_channels(values)
    x = torch.from_numpy(
        np.frombuffer(b"".join(values), dtype=np.uint8).reshape(len(values), channels, 8, 8).copy()
    ).float() / 255.0
    x = narrow_board_planes(x, min(source_board_channels, channels // 2))
    return widen_board_planes(x, board_channels)


def drift_outputs(model: nn.Module, x: Tensor, ctx, batch_size: int = 256):
    """(value (N,), log-softmax policy A (N, 4672), policy B) of model on x."""
    board = getattr(model, "board_channels", None) or x.shape[1] // 2
    was_training = model.training
    model.eval()
    values, logp_a, logp_b = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]
            batch = narrow_board_planes(batch, board) if batch.shape[1] // 2 > board else \
                widen_board_planes(batch, board)
            out = model(batch.to(ctx))
            values.append(torch.flatten(out[0]).float().cpu())
            logp_a.append(torch.log_softmax(out[1][0].float(), dim=1).cpu())
            logp_b.append(torch.log_softmax(out[1][1].float(), dim=1).cpu())
    if was_training:
        model.train()
    return torch.cat(values), torch.cat(logp_a), torch.cat(logp_b)


def drift_metrics(reference, current) -> dict:
    """policy_kl: KL(reference || current) per board, averaged over boards and
    positions; value_mse: mean squared value difference."""
    ref_v, ref_a, ref_b = reference
    cur_v, cur_a, cur_b = current
    kl = 0.0
    for ref, cur in ((ref_a, cur_a), (ref_b, cur_b)):
        kl += float((ref.exp() * (ref - cur)).sum(dim=1).mean())
    return {
        "policy_kl": kl / 2.0,
        "value_mse": float(((cur_v - ref_v) ** 2).mean()),
        "positions": int(len(ref_v)),
    }


def new_columns_max_abs(model: nn.Module, source_board_channels: int = 37) -> float:
    """Largest |weight| over the widened input columns (0 when there are none)."""
    masks = new_input_columns(model, source_board_channels)
    params = dict(model.named_parameters())
    best = 0.0
    for name, mask in masks.items():
        if bool(mask.any()):
            best = max(best, float(params[name].detach()[:, mask].abs().max()))
    return best
