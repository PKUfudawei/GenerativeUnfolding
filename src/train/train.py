from typing import Optional
from collections import defaultdict
from tqdm import tqdm
import time
from datetime import timedelta
import os
import torch
from ema_pytorch import EMA
from ..models.inn import INN
from ..models.transfermer import Transfermer
from ..models.transfermer_1d import Transfermer1d
from ..models.cfm import CFM, CFMwithTransformer, TransfusionAR
from ..models.didi import DirectDiffusion, DirectDiffusion_Padded
from ..models.classifier import Classifier
from ..models.fff import FreeFormFlow
from .preprocessing import build_preprocessing, PreprocChain
from ..processes.base import Process, ProcessData
from .documenter import Documenter
from ..processes.zjets.process import ZJetsGenerative, ZJetsOmnifold


class Model:
    """
    Class for training, evaluating, loading and saving models for density estimation or
    importance sampling.
    """

    def __init__(
        self,
        params: dict,
        verbose: bool,
        device: torch.device,
        model_path: str,
        state_dict_attrs: list[str],
    ):
        """
        Initializes the training class.

        Args:
            params: Dictionary with run parameters
            verbose: print
            device: pytorch device
            model_path: path to save trained models and checkpoints in
            input_data: tensors with training, validation and test input data
            cond_data: tensors with training, validation and test condition data
            state_dict_attrs: list of attribute whose state dicts will be stored
        """
        self.params = params
        self.device = device
        self.model_path = model_path
        self.verbose = verbose
        self.is_classifier = False
        model = params.get("model", "INN")
        print(f"    Model class: {model}")
        try:
            self.model = eval(model)(params)
        except NameError:
            print(model)
            raise NameError("model not recognised. Use exact class name")
        self.model.to(device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Total trainable parameters: {n_params}")
        #if hasattr(torch, "compile"):
        #    print("  Compiling model")
        #    self.mode = torch.compile(self.model)
        self.state_dict_attrs = [*state_dict_attrs, "model", "optimizer"]
        self.losses = defaultdict(list)

    def init_data_loaders(
        self,
        input_data: tuple[torch.Tensor, ...],
        cond_data: tuple[torch.Tensor, ...],
    ):
        input_train, input_val, input_test = input_data
        cond_train, cond_val, cond_test = cond_data
        self.n_train_samples = len(input_train)
        self.n_val_samples = len(input_val)
        self.bs = self.params.get("batch_size")
        self.bs_sample = self.params.get("batch_size_sample", self.bs)
        train_loader_kwargs = {"shuffle": True, "batch_size": self.bs, "drop_last": False}
        val_loader_kwargs = {"shuffle": False, "batch_size": self.bs_sample, "drop_last": False}

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input_train.float(), cond_train.float()),
            **train_loader_kwargs,
        )
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input_val.float(), cond_val.float()),
            **val_loader_kwargs,
        )
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input_test.float(), cond_test.float()),
            **val_loader_kwargs,
        )

    def progress(self, iterable, **kwargs):
        """
        Shows a progress bar if verbose training is enabled

        Args:
            iterable: iterable object
            kwargs: keyword arguments passed on to tqdm if verbose
        Returns:
            Unchanged iterable if not verbose, otherwise wrapped by tqdm
        """
        if self.verbose:
            return tqdm(iterable, **kwargs)
        else:
            return iterable

    def print(self, text):
        """
        Chooses print function depending on verbosity setting

        Args:
            text: String to be printed
        """
        if self.verbose:
            tqdm.write(text)
        else:
            print(text, flush=True)

    def init_optimizer(self):
        """
        Initialize optimizer and learning rate scheduling
        """
        optimizer = {
            "adam": torch.optim.Adam,
            "radam": torch.optim.RAdam,
        }[self.params.get("optimizer", "adam")]
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=self.params.get("lr", 0.0002),
            betas=self.params.get("betas", [0.9, 0.999]),
            eps=self.params.get("eps", 1e-6),
            weight_decay=self.params.get("weight_decay", 0.0),
        )

        self.lr_sched_mode = self.params.get("lr_scheduler", None)
        if self.lr_sched_mode == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.params["lr_decay_epochs"],
                gamma=self.params["lr_decay_factor"],
            )
        elif self.lr_sched_mode == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                self.params.get("max_lr", self.params["lr"] * 10),
                epochs=self.params["epochs"],
                steps_per_epoch=len(self.train_loader),
            )
        elif self.lr_sched_mode == "cosine_annealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max= self.params["epochs"] * len(self.train_loader)
            )
        else:
            self.scheduler = None

        self.use_ema = self.params.get("use_ema", False)
        if self.use_ema:
            self.model.use_ema = True
            n_epochs = self.params.get("n_epochs", 100)
            trainloader_length = len(self.train_loader)
            ema_start = self.params.get("ema_start", 0.8)
            ema_start_iter = int(ema_start * n_epochs * trainloader_length)
            self.model.ema = EMA(self.model.net, update_after_step=ema_start_iter, update_every=10).to(self.device)
            print(f"    Using EMA with start at {ema_start}")

    def begin_epoch(self):
        """
        Overload this function to perform some task at the beginning of each epoch.
        """
        pass

    def train(self):
        """
        Main training loop
        """
        best_val_loss = 1e20
        checkpoint_interval = self.params.get("checkpoint_interval")
        checkpoint_overwrite = self.params.get("checkpoint_overwrite", True)
        use_ema = self.params.get("use_ema", False)

        start_time = time.time()
        for epoch in self.progress(
            range(self.params["epochs"]), desc="  Epoch", leave=False, position=0
        ):
            self.begin_epoch()
            self.model.train()
            epoch_train_losses = defaultdict(int)
            loss_scale = 1 / len(self.train_loader)
            for xs, cs in self.progress(
                self.train_loader, desc="  Batch", leave=False, position=1
            ):
                self.optimizer.zero_grad()
                loss, loss_terms = self.model.batch_loss(
                    xs, cs, 1 / self.n_train_samples
                )
                loss.backward()
                self.optimizer.step()
                if self.lr_sched_mode == "one_cycle" or self.lr_sched_mode == "cosine_annealing":
                    self.scheduler.step()
                for name, loss in loss_terms.items():
                    epoch_train_losses[name] += loss * loss_scale
                if use_ema:
                    self.model.ema.update()
            if self.lr_sched_mode == "step":
                self.scheduler.step()

            for name, loss in epoch_train_losses.items():
                self.losses[f"tr_{name}"].append(loss)
            for name, loss in self.dataset_loss(self.val_loader).items():
                self.losses[f"val_{name}"].append(loss)
            if epoch < 20:
                last_20_val_losses = self.losses["val_loss"]
            else:
                last_20_val_losses = self.losses["val_loss"][-20:]
            self.losses["val_movAvg"].append(torch.tensor(last_20_val_losses).mean().item())

            self.losses["lr"].append(self.optimizer.param_groups[0]["lr"])

            if self.losses["val_loss"][-1] < best_val_loss:
                best_val_loss = self.losses["val_loss"][-1]
                self.save("best")
            if (
                checkpoint_interval is not None
                and (epoch + 1) % checkpoint_interval == 0
            ):
                self.save("final" if checkpoint_overwrite else f"epoch_{epoch}")

            self.print(
                f"    Ep {epoch}: "
                + ", ".join(
                    [
                        f"{name} = {loss[-1]:{'.2e' if name == 'lr' else '.5f'}}"
                        for name, loss in self.losses.items()
                    ]
                )
                + f", t = {timedelta(seconds=round(time.time() - start_time))}"
            )

        self.save("final")
        time_diff = timedelta(seconds=round(time.time() - start_time))
        print(f"    Training completed after {time_diff}")

    def dataset_loss(self, loader: torch.utils.data.DataLoader) -> dict:
        """
        Computes the losses (without gradients) for the given data loader

        Args:
            loader: data loader
        Returns:
            Dictionary with loss terms averaged over all samples
        """
        self.model.eval()
        if self.model.bayesian:
            self.model.reset_random_state()
        n_total = 0
        total_losses = defaultdict(list)
        with torch.no_grad():
            for xs, cs in self.progress(loader, desc="  Batch", leave=False):
                n_samples = xs.shape[0]
                n_total += n_samples
                _, losses = self.model.batch_loss(
                    xs, cs, kl_scale=1 / self.n_train_samples
                )
                for name, loss in losses.items():
                    total_losses[name].append(loss * n_samples)
        return {name: sum(losses) / n_total for name, losses in total_losses.items()}

    def predict(self, loader=None) -> torch.Tensor:
        """
        Predict one sample for each event from the test data

        Returns:
            tensor with samples, shape (n_events, dims_in)
        """
        self.model.eval()
        bayesian_samples = self.params.get("bayesian_samples", 20) if self.model.bayesian else 1
        n_unfoldings = self.params.get("n_unfoldings", 1)

        if loader is None:
            loader = self.test_loader

        with torch.no_grad():
            all_samples = []
            for i in range(bayesian_samples):
                t0 = time.time()
                unfoldings = []
                if self.model.bayesian:
                    if i == 0:
                        for layer in self.model.bayesian_layers:
                            layer.map = True
                    else:
                        for layer in self.model.bayesian_layers:
                            layer.map = False
                        self.model.reset_random_state()

                for j in range(n_unfoldings):
                    data_batches = []
                    for xs, cs in self.progress(
                        loader,
                        desc="  Generating",
                        leave=False,
                        initial=i * len(loader),
                        total=bayesian_samples * len(loader),
                    ):
                        while True:
                            try:
                                data_batches.append(self.model.sample(cs))
                                #data_batches.append(xs)
                                break
                            except AssertionError:
                                print(f"    Batch failed, repeating")
                    unfoldings.append(torch.cat(data_batches, dim=0))
                unfoldings = torch.stack(unfoldings, dim=0)
                all_samples.append(unfoldings)
                if self.model.bayesian:
                    print(f"    Finished bayesian sample {i} in {time.time() - t0}", flush=True)
            all_samples = torch.stack(all_samples, dim=0)
            if self.model.bayesian:
                return all_samples#.reshape(bayesian_samples, -1, 6)
            else:
                return all_samples[0]#.reshape(-1, 6)

    def predict_distribution(self, loader=None) -> torch.Tensor:
        """
        Predict multiple samples for a part of the test dataset

        Returns:
            tensor with samples, shape (n_events, n_samples, dims_in)
        """
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        max_batches = min(len(loader), self.params.get("max_dist_batches", 1))
        samples_per_event = self.params.get("dist_samples_per_event", 5)
        if self.model.bayesian:
            for layer in self.model.bayesian_layers:
                layer.map = True

        with torch.no_grad():
            all_samples = []
            for i, (xs, cs) in enumerate(loader):
                if i == max_batches:
                    break
                data_batches = []
                for _ in self.progress(
                    range(samples_per_event),
                    desc="  Generating",
                    leave=False,
                ):
                    while True:
                        try:
                            data_batches.append(self.model.sample(cs))
                            break
                        except AssertionError:
                            print("Batch failed, repeating")
                all_samples.append(torch.stack(data_batches, dim=1))
            all_samples = torch.cat(all_samples, dim=0)
            return all_samples

    def save(self, name: str):
        """
        Saves the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        torch.save(
            {
                **{
                    attr: getattr(self, attr).state_dict()
                    for attr in self.state_dict_attrs
                },
                "losses": self.losses,
            },
            file,
        )
        if self.use_ema:
            file = os.path.join(self.model_path, f"ema_{name}.pth")
            torch.save(self.model.ema.state_dict(), file)

    def load(self, name: str):
        """
        Loads the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        state_dicts = torch.load(file, map_location=self.device)
        for attr in self.state_dict_attrs:
            try:
                getattr(self, attr).load_state_dict(state_dicts[attr])
            except AttributeError:
                pass
        self.losses = state_dicts["losses"]

        if self.params.get("use_ema", False):
            self.use_ema = True
            self.model.use_ema = True
            file = os.path.join(self.model_path, f"ema_{name}.pth")
            ema_dict = torch.load(file, map_location=self.device)
            self.model.ema = EMA(self.model.net).to(self.device)
            self.model.ema.load_state_dict(ema_dict)


class GenerativeUnfolding(Model):
    def __init__(
        self,
        params: dict,
        verbose: bool,
        device: torch.device,
        model_path: str,
        process: Process
    ):
        self.process = process

        self.hard_pp = build_preprocessing(params.get("hard_preprocessing", {}))
        self.reco_pp = build_preprocessing(params.get("reco_preprocessing", {}))
        self.hard_pp.to(device)
        self.reco_pp.to(device)

        params["dims_in"] = self.hard_pp.output_shape[0]
        params["dims_c"] = self.reco_pp.output_shape[0]


        super().__init__(
            params,
            verbose,
            device,
            model_path,
            state_dict_attrs=["hard_pp", "reco_pp"],
        )

        self.unpaired = params.get("unpaired", False)
        if self.unpaired:
            assert isinstance(self.model, DirectDiffusion)
            print(f"    Using unpaired data")

    def init_data_loaders(self):
        data = (
        self.process.get_data("train"),
        self.process.get_data("val"),
        self.process.get_data("test"),
        )
        if self.params.get("joint_normalization", False):
            self.hard_pp.init_normalization(data[0].x_hard)
            self.reco_pp.init_normalization(data[0].x_hard)
        else:
            self.hard_pp.init_normalization(data[0].x_hard)
            self.reco_pp.init_normalization(data[0].x_reco)
        self.input_data_preprocessed = tuple(self.hard_pp(subset.x_hard) for subset in data)
        self.cond_data_preprocessed = tuple(self.reco_pp(subset.x_reco) for subset in data)
        print(f"    Preprocessed: Hard train shape {self.input_data_preprocessed[0].shape}, Reco train shape {self.cond_data_preprocessed[0].shape}")
        super(GenerativeUnfolding, self).init_data_loaders(self.input_data_preprocessed, self.cond_data_preprocessed)

    def begin_epoch(self):
        # The only difference between paired and unpaired is shuffling the condition data each epoch
        if not self.unpaired:
            return

        train_loader_kwargs = {"shuffle": True, "batch_size": self.bs, "drop_last": False}
        val_loader_kwargs = {"shuffle": False, "batch_size": self.bs_sample, "drop_last": False}

        input_train = self.input_data_preprocessed[0].clone()
        cond_train = self.cond_data_preprocessed[0].clone()
        input_val = self.input_data_preprocessed[1].clone()
        cond_val = self.cond_data_preprocessed[1].clone()

        permutation_train = torch.randperm(self.n_train_samples)
        permutation_val = torch.randperm(self.n_val_samples)
        cond_train = cond_train[permutation_train]
        cond_val = cond_val[permutation_val]

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input_train.float(), cond_train.float()),
            **train_loader_kwargs,
        )
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input_val.float(), cond_val.float()),
            **val_loader_kwargs,
        )

    def predict(self, loader=None) -> torch.Tensor:
        """
        Predict one sample for each event in the test dataset

        Returns:
            tensor with samples, shape (n_events, dims_in)
        """
        samples = super().predict(loader)
        if self.model.bayesian:
            samples_pp = torch.zeros((samples.shape[0], samples.shape[1], samples.shape[2], *self.hard_pp.input_shape))
            for bayesian_sample in range(samples.shape[0]):
                for unfolding in range(samples.shape[1]):
                    samples_pp[bayesian_sample, unfolding] = self.hard_pp(samples[bayesian_sample, unfolding], rev=True)
        else:
            samples_pp = torch.zeros((samples.shape[0], samples.shape[1], *self.hard_pp.input_shape))
            for unfolding in range(samples.shape[0]):
                samples_pp[unfolding] = self.hard_pp(samples[unfolding], rev=True)
        return samples_pp

    def predict_distribution(self, loader=None) -> torch.Tensor:
        """
        Predict multiple samples for a part of the test dataset

        Returns:
            tensor with samples, shape (n_events, n_samples, dims_in)
        """
        samples = super().predict_distribution(loader)
        samples_pp = self.hard_pp(
            samples.reshape(-1, samples.shape[-1]),
            rev=True,
            batch_size=1000,
        )
        return samples_pp.reshape(*samples.shape[:2], *samples_pp.shape[1:])


class Omnifold(Model):
    def __init__(
        self,
        params: dict,
        verbose: bool,
        device: torch.device,
        model_path: str,
        process: Process
    ):
        self.process = process

        self.hard_pp = build_preprocessing(params["hard_preprocessing"], n_dim=params["dims_c"])
        self.reco_pp = build_preprocessing(params["reco_preprocessing"], n_dim=params["dims_in"])
        self.hard_pp.to(device)
        self.reco_pp.to(device)
        params["dims_in"] = self.hard_pp.output_shape[0]
        params["dims_c"] = self.reco_pp.output_shape[0]

        super().__init__(
            params,
            verbose,
            device,
            model_path,
            state_dict_attrs=["hard_pp", "reco_pp"],
        )

    def init_data_loaders(self):
        data = (
        self.process.get_data("train"),
        self.process.get_data("val"),
        self.process.get_data("test"),
        )
        label_data = tuple(subset.label for subset in data)
        self.reco_pp.init_normalization(data[0].x_reco)
        reco_data = tuple(self.reco_pp(subset.x_reco) for subset in data)

        super(Omnifold, self).init_data_loaders(label_data, reco_data)

    def predict_probs(self, loader=None):
        self.model.eval()

        if loader is None:
            loader = self.test_loader

        bayesian_samples = self.params.get("bayesian_samples", 20) if self.model.bayesian else 1
        with torch.no_grad():
            all_samples = []
            for i in range(bayesian_samples):
                if self.model.bayesian:
                    if i == 0:
                        for layer in self.model.bayesian_layers:
                            layer.map = True
                    else:
                        for layer in self.model.bayesian_layers:
                            layer.map = False
                        self.model.reset_random_state()
                predictions = []
                t0 = time.time()
                for xs, cs in self.progress(loader, desc="  Predicting", leave=False):
                    predictions.append(self.model.probs(cs))
                all_samples.append(torch.cat(predictions, dim=0))
                if self.model.bayesian:
                    print(f"    Finished bayesian sample {i} in {time.time() - t0}", flush=True)
            all_samples = torch.cat(all_samples, dim=0)
        if self.model.bayesian:
            return all_samples.reshape(
                bayesian_samples,
                len(all_samples) // bayesian_samples,
                *all_samples.shape[1:],
            )
        else:
            return all_samples

