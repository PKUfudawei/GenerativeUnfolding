import warnings
from dataclasses import dataclass
from typing import Optional
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import torch
from .utils import GetEMD, get_triangle_distance, get_normalisation_weight
from ..processes.observables import Observable
import yaml
from scipy.stats import binned_statistic


@dataclass
class Line:
    y: np.ndarray
    y_err: Optional[np.ndarray] = None
    y_ref: Optional[np.ndarray] = None
    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: str = "solid"
    fill: bool = False
    vline: bool = False


class Plots:
    """
    Implements the plotting pipeline to evaluate the performance of conditional generative
    networks.
    """

    def __init__(
        self,
        observables: list[Observable],
        losses: dict,
        x_hard: torch.Tensor,
        x_reco: torch.Tensor,
        x_gen_single: torch.Tensor,
        x_gen_dist: torch.Tensor,
        x_compare=None,
        x_hard_pp=None,
        x_reco_pp=None,
        bayesian: bool = False,
        show_metrics: bool = True,
        plot_metrics: bool = False
    ):
        """
        Initializes the plotting pipeline with the data to be plotted.
        Args:
            doc: Documenter object
            observables: List of observables
            losses: Dictionary with loss terms and learning rate as a function of the epoch
            x_test: Data from test dataset
            x_gen_single: Generated data (single point for each test sample)
            x_gen_dist: Generated data (multiple points for subset of test data)
            event_type: Event types for the test dataset
            bayesian: Boolean that indicates if the model is bayesian
        """
        self.observables = observables
        self.losses = losses

        self.compare = x_compare is not None

        self.x_hard_pp = x_hard_pp.cpu().numpy() if x_hard_pp is not None else None
        self.x_reco_pp = x_reco_pp.cpu().numpy() if x_reco_pp is not None else None

        self.x_hard = x_hard.cpu().numpy()
        self.x_reco = x_reco.cpu().numpy()

        self.obs_hard = []
        self.obs_reco = []
        self.obs_gen_single = []
        self.obs_gen_dist = []
        self.obs_compare = []
        self.bins = []
        for obs in observables:
            o_hard = obs.compute(x_hard)
            self.obs_hard.append(o_hard.cpu().numpy())

            o_reco = obs.compute(x_reco)
            self.obs_reco.append(o_reco.cpu().numpy())

            o_gen_single = obs.compute(x_gen_single)
            self.obs_gen_single.append(o_gen_single.cpu().numpy())

            o_gen_dist = obs.compute(x_gen_dist)
            self.obs_gen_dist.append(o_gen_dist.cpu().numpy())

            self.bins.append(obs.bins(o_hard).cpu().numpy())

            if self.compare:
                o_compare = obs.compute(x_compare).cpu().numpy()
            else:
                o_compare = None
            self.obs_compare.append(o_compare)

        self.n_unfoldings = x_gen_single.shape[-3]
        self.bayesian = bayesian
        self.show_metrics = show_metrics
        self.colors = [f"C{i}" for i in range(20)]
        self.plot_metrics = plot_metrics
        if self.show_metrics:
            print(f"    Computing metrics")
            self.compute_metrics()

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        #plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=False)

    def plot_losses(self, file: str):
        """
        Makes plots of the losses (total loss and if bayesian, BCE loss and KL loss
        separately) and learning rate as a function of the epoch.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            self.plot_single_loss(
                pp,
                "loss",
                (self.losses["train_loss"], self.losses["val_loss"]),
                ("train", "val"),
            )
            if self.bayesian:
                self.plot_single_loss(
                    pp,
                    "INN loss",
                    (self.losses["train_inn_loss"], self.losses["val_inn_loss"]),
                    ("train", "val"),
                )
                self.plot_single_loss(
                    pp,
                    "KL loss",
                    (self.losses["train_kl_loss"], self.losses["val_kl_loss"]),
                    ("train", "val"),
                )
            self.plot_single_loss(
                pp, "learning rate", (self.losses["lr"],), (None,), "log"
            )

    def plot_single_loss(
        self,
        pp: PdfPages,
        ylabel: str,
        curves: tuple[np.ndarray],
        labels: tuple[str],
        yscale: str = "linear",
    ):
        """
        Makes single loss plot.
        Args:
            pp: Multipage PDF object
            ylabel: Y axis label
            curves: List of numpy arrays with the loss curves to be plotted
            labels: Labels of the loss curves
            yscale: Y axis scale, "linear" or "log"
        """
        fig, ax = plt.subplots(figsize=(4, 3.5))
        for i, (curve, label) in enumerate(zip(curves, labels)):
            epochs = np.arange(6, len(curve) + 6)
            ax.plot(epochs[5:], curve[5:], label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        if any(label is not None for label in labels):
            ax.legend(loc="center right", frameon=False)
        plt.savefig(pp, format="pdf", bbox_inches="tight")
        plt.close()

    def plot_observables(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare
            ):
                if self.bayesian:
                    data = data_gen[:, 0]
                else:
                    data = data_gen[0]

                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_reco, y_reco_err = self.compute_hist_data(bins, data_reco, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data, bayesian=self.bayesian)
                lines = [
                    Line(
                        y=y_reco,
                        y_err=y_reco_err,
                        label="Reco",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label="Unfold",
                        color=self.colors[1],
                    ),
                ]
                if self.bayesian:
                    bay_n, dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(bay_n, -1)
                else:
                    dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(-1)
                y_gen_full, y_gen_err = self.compute_hist_data(bins, data, bayesian=self.bayesian)

                lines.append(
                    Line(
                        y=y_gen_full,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label=f"{dist_n} Unfoldings",
                        color=self.colors[3]
                    )
                )

                if self.compare:
                    y_comp, y_comp_err = self.compute_hist_data(bins, data_compare, bayesian=False)
                    lines.append(
                        Line(
                            y=y_comp,
                            y_err=y_comp_err,
                            y_ref=y_hard,
                            label="SB",
                            color=self.colors[3]
                        )
                    )
                if not self.bayesian:
                    metrics = [obs.metrics["emd_arr"][0][0],
                               obs.metrics["emd_arr"][0].std(),
                               obs.metrics["triangle_arr"][0][0],
                               obs.metrics["triangle_arr"][0].std()]
                else:
                    metrics = [obs.metrics["emd_arr"][0][0],
                               obs.metrics["emd_arr"][:, 0].std(),
                               obs.metrics["triangle_arr"][0][0],
                               obs.metrics["triangle_arr"][:, 0].std()]
                self.hist_plot(pp, lines, bins, obs, metrics=metrics)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_calibration(self, file: str):
        """
        Makes calibration plots for all observables.

        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, data_true, data_gen in zip(
                self.observables, self.obs_hard, self.obs_gen_dist
            ):
                data_true = data_true[None, ...]
                data_gen = data_gen[None, ...]
                data_true = data_true[:, : data_gen.shape[1]]
                calibration_x = np.linspace(0, 1, 101)
                for i, (data_true_elem, data_gen_elem) in enumerate(
                    zip(data_true, data_gen)
                ):
                    quantiles = np.mean(data_gen_elem < data_true_elem[:, None], axis=1)
                    calibration_y = np.mean(
                        quantiles[:, None] < calibration_x[None, :], axis=0
                    )
                    plt.plot(
                        calibration_x,
                        calibration_y,
                        color=self.colors[0],
                        alpha=0.3 if self.bayesian else 1,
                    )

                plt.plot([0, 1], [0, 1], color="k", linestyle=":")
                plt.xlabel(f"quantile ${{{obs.tex_label}}}$")
                plt.ylabel("fraction of events")
                plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
                plt.close()

    def plot_pulls(self, file: str):
        """
        Plots pulls for all observables.
        Args:
            file: Output file name
        """

        bins = np.linspace(-5, 5, 40)
        with PdfPages(file) as pp:
            for obs, data_true, data_gen in zip(
                self.observables, self.obs_hard, self.obs_gen_dist
            ):
                data_true = data_true[: data_gen.shape[-2]]
                gen_mean = np.mean(data_gen, axis=-1)
                gen_std = np.std(data_gen, axis=-1)
                pull = (gen_mean - data_true) / gen_std
                y, y_err = self.compute_hist_data(bins, pull)
                lines = [
                    Line(
                        y=y,
                        y_err=y_err,
                        label="Pull",
                        color=self.colors[0],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs, show_ratios=False)

    def plot_single_events(self, file: str):
        """
        Plots single event distributions for all observables.
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for j, (obs, bins, data_true, data_reco, data_gen) in enumerate(zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_dist
            )):
                for i in range(5):
                    y_gen, y_gen_err = self.compute_hist_data(bins, data_gen[..., i, :])
                    y_hard, y_hard_err = self.compute_hist_data(bins, data_true, bayesian=False)
                    n_unfoldings = data_gen[..., i, :].shape[-1]

                    # extract reco event in plot dimension
                    event_reco_plotdim = data_reco[i]

                    # calculate close reco events in plot dimension
                    std_j = self.x_reco[:, j].std(0)
                    mask_plotdim = np.abs(self.x_reco[:, j] - event_reco_plotdim) < 0.001*std_j
                    n_close_events = mask_plotdim.sum()
                    print(f"    dim {j}, event {i}, close events {n_close_events}")
                    #n_close_events = 400
                    #mask_plotdim = np.argsort(np.abs(self.x_reco[:, j] - event_reco_plotdim))[:n_close_events]

                    # extract reco and hard events that are close in plot reco dimension
                    close_events_reco_plotdim = self.x_reco[mask_plotdim]
                    close_events_hard_plotdim = self.x_hard[mask_plotdim]

                    # calculate histograms
                    y_gt, _ = self.compute_hist_data(bins, close_events_hard_plotdim[:, j])
                    y_close_reco, _ = self.compute_hist_data(bins, close_events_reco_plotdim[:, j])

                    # plot
                    lines = [
                        Line(
                            y=data_true[i],
                            label="Truth",
                            color=self.colors[1],
                            vline=True,
                        ),
                        Line(
                            y=event_reco_plotdim,
                            label="Reco",
                            color=self.colors[4],
                            vline=True,
                        ),
                        Line(
                            y=y_hard,
                            label="Full hard",
                            color=self.colors[7],
                        ),
                        Line(
                            y=y_gen,
                            y_ref=None,
                            label="Gen",
                            color=self.colors[0],
                        ),
                        Line(
                            y=y_gt,
                            y_ref=None,
                            label="GT",
                            color=self.colors[2],
                        )
                    ]
                    self.hist_plot(
                        pp, lines, bins, obs, title=f"Event {i}", show_ratios=False
                    )

                    # cut on other 6 dimensions
                    for cut_dim in range(6):
                        # skip if cut dimension is plot dimension
                        if cut_dim == j:
                            continue

                        # extract close reco events in cut dimension
                        close_events_reco_cut_dim = close_events_reco_plotdim[:, cut_dim]
                        # find out where the test event is relative to other events in cut_dim
                        reco_cutdim = self.obs_reco[cut_dim]
                        event_reco_cutdim = self.obs_reco[cut_dim][i]
                        #mask_cutdim = np.argsort(np.abs(self.x_reco[:, cut_dim] - event_reco_cutdim))[:n_close_events]

                        # std_cut_dim = self.x_reco[:, cut_dim].std(0)
                        # mask_cutdim = np.abs(self.x_reco[:, cut_dim] - event_reco_cutdim) < 0.1 * std_cut_dim
                        #n_close_events = mask_plotdim.sum()

                        # full_mask = mask_plotdim*mask_cutdim

                        # ct on cutdim
                        # close_events_hard_plotdim_cutdim = self.x_hard[full_mask]

                        where_in_cutdim = (close_events_reco_cut_dim < event_reco_cutdim).mean()
                        # split close reco events in halfs sorted by cut dimension
                        close_events_reco_cut_dim = np.argsort(close_events_reco_cut_dim)

                        quantile25_1 = close_events_reco_cut_dim[:int(n_close_events/4)]
                        quantile25_2 = close_events_reco_cut_dim[int(n_close_events / 4):2*int(n_close_events / 4)]
                        quantile25_3 = close_events_reco_cut_dim[int(n_close_events / 4)*2:3 * int(n_close_events / 4)]
                        quantile25_4 = close_events_reco_cut_dim[3*int(n_close_events / 4):]

                        # half_1, half_2 = close_events_reco_cut_dim[:int(n_close_events/2)], close_events_reco_cut_dim[int(n_close_events/2):]
                        # make histogram of corresponding hard events in plot dimension



                        # make histograms of upper and lower 50quantile in cutdim
                        # y_gt_1, y_gt_err = self.compute_hist_data(bins, close_events_hard_plotdim[half_1, j])
                        # y_gt_2, y_gt_err = self.compute_hist_data(bins, close_events_hard_plotdim[half_2, j])

                        # make histograms of 25quantiles in cutdim
                        y_gt_quant1, y_gt_err = self.compute_hist_data(bins, close_events_hard_plotdim[quantile25_1, j])
                        y_gt_quant2, y_gt_err = self.compute_hist_data(bins, close_events_hard_plotdim[quantile25_2, j])
                        y_gt_quant3, y_gt_err = self.compute_hist_data(bins, close_events_hard_plotdim[quantile25_3, j])
                        y_gt_quant4, y_gt_err = self.compute_hist_data(bins, close_events_hard_plotdim[quantile25_4, j])

                        # make histogram of twice cut
                        # y_gt_cut, _ = self.compute_hist_data(bins, close_events_hard_plotdim_cutdim[:, j])

                        lines = [
                            # Line(
                            #     y=data_true[i],
                            #     label="Truth",
                            #     color=self.colors[1],
                            #     vline=True,
                            # ),
                            # Line(
                            #     y=event_reco,
                            #     label="Reco",
                            #     color=self.colors[4],
                            #     vline=True,
                            # ),
                             Line(
                                 y=y_gen,
                                 y_ref=None,
                                 label="Gen",
                                 color=self.colors[0],
                             ),
                             # Line(
                             #     y=y_gt,
                             #     y_ref=None,
                             #     label="GT",
                             #     color=self.colors[2],
                             # ),
                             # Line(
                             #     y=y_gt_cut,
                             #     y_ref=None,
                             #     label=f"GT cut {cut_dim}",
                             #     color=self.colors[9],
                             # ),
                            # Line(
                            #    y=y_gt_1,
                            #    y_ref=None,
                            #    label="GT below med",
                            #    color=self.colors[5],
                            # ),
                            # Line(
                            #    y=y_gt_2,
                            #    y_ref=None,
                            #    label="GT above med",
                            #    color=self.colors[6],
                            # )
                            Line(
                                y=y_gt_quant1,
                                y_ref=None,
                                label="GT 25Quant 1",
                                color=self.colors[11],
                            ),
                            # Line(
                            #     y=y_gt_quant2,
                            #     y_ref=None,
                            #     label="GT 25Quant 2",
                            #     color=self.colors[12],
                            # ),
                            # Line(
                            #     y=y_gt_quant3,
                            #     y_ref=None,
                            #     label="GT 25Quant 3",
                            #     color=self.colors[13],
                            # ),
                            Line(
                                y=y_gt_quant4,
                                y_ref=None,
                                label="GT 25Quant 4",
                                color=self.colors[14],
                            ),
                        ]
                        #self.hist_plot(
                        #    pp, lines, bins, obs, title=f"Event {i} vs dim {cut_dim}, pos {where_in_cutdim:.2f}", show_ratios=False
                        #)

    def plot_preprocessed(self, file: str):
        """
        Plots preprocessed observable distributions
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, data_hard_pp, data_reco_pp in zip(
                    self.observables, self.x_hard_pp.T, self.x_reco_pp.T
            ):
                bins = 100
                y_hard, bins = np.histogram(data_hard_pp.cpu().numpy(), bins=bins, density=True)
                y_reco, _ = np.histogram(data_reco_pp.cpu().numpy(), bins=bins, density=True)
                #y_gen, y_gen_err = self.compute_hist_data(bins, data_gen, bayesian=self.bayesian)
                normal = np.random.normal(size=data_hard_pp.shape)
                y_normal, _ = np.histogram(normal, bins=bins, density=True)
                lines = [
                    Line(
                        y=y_reco,
                        y_err=None,
                        label="Reco PP",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=None,
                        label="Hard PP",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_normal,
                        y_err=None,
                        label="Normal",
                        color=self.colors[5],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs, metrics=None, show_ratios=False, no_scale=True)

    def compute_hist_data(self, bins: np.ndarray, data: np.ndarray, bayesian=False, weights=None):
        if bayesian:
            hists = np.stack(
                [np.histogram(d, bins=bins, density=True, weights=weights)[0] for d in data], axis=0
            )
            y = hists[0]
            y_err = np.std(hists, axis=0)
        else:
            y, _ = np.histogram(data, bins=bins, density=False, weights=weights)
            y_err = np.sqrt(y)
        return y, y_err

    def hist_plot(
        self,
        pdf: PdfPages,
        lines: list[Line],
        bins: np.ndarray,
        observable: Observable,
        show_ratios: bool = True,
        title: Optional[str] = None,
        no_scale: bool = False,
        yscale: Optional[str] = None,
        metrics = None
    ):
        """
        Makes a single histogram plot, used for the observable histograms and clustering
        histograms.
        Args:
            pdf: Multipage PDF object
            lines: List of line objects describing the histograms
            bins: Numpy array with the bin boundaries
            show_ratios: If True, show a panel with ratios
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            n_panels = 1 + int(show_ratios) + int(metrics is not None)
            fig, axs = plt.subplots(
                n_panels,
                1,
                sharex=True,
                figsize=(6, 4.5),
                gridspec_kw={"height_ratios": (12, 3, 1)[:n_panels], "hspace": 0.00},
            )
            if n_panels == 1:
                axs = [axs]

            for line in lines:
                if line.vline:
                    axs[0].axvline(line.y, label=line.label, color=line.color, linestyle=line.linestyle)
                    continue
                integral = np.sum((bins[1:] - bins[:-1]) * line.y)
                scale = 1 / integral if integral != 0.0 else 1.0
                if line.y_ref is not None:
                    ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                    ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
                if no_scale:
                    scale = 1.
                    ref_scale = 1.

                self.hist_line(
                    axs[0],
                    bins,
                    line.y * scale,
                    line.y_err * scale if line.y_err is not None else None,
                    label=line.label,
                    color=line.color,
                    fill=line.fill,
                    linestyle=line.linestyle
                )

                if line.y_ref is not None:
                    ratio = (line.y * scale) / (line.y_ref * ref_scale)
                    ratio_isnan = np.isnan(ratio)
                    if line.y_err is not None:
                        if len(line.y_err.shape) == 2:
                            ratio_err = (line.y_err * scale) / (line.y_ref * ref_scale)
                            ratio_err[:, ratio_isnan] = 0.0
                        else:
                            ratio_err = np.sqrt((line.y_err / line.y) ** 2)
                            ratio_err[ratio_isnan] = 0.0
                    else:
                        ratio_err = None
                    ratio[ratio_isnan] = 1.0
                    self.hist_line(
                        axs[1], bins, ratio, ratio_err, label=None, color=line.color
                    )

            axs[0].legend(frameon=False)
            axs[0].set_ylabel("normalized")
            axs[0].set_yscale(observable.yscale if yscale is None else yscale)
            if title is not None:
                self.corner_text(axs[0], title, "left", "top")

            if show_ratios:
                axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Truth}}$")
                axs[1].set_yticks([0.95, 1, 1.05])
                axs[1].set_ylim([0.9, 1.1])
                axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
                axs[1].axhline(y=1.05, c="black", ls="dotted", lw=0.5)
                axs[1].axhline(y=0.95, c="black", ls="dotted", lw=0.5)

            if metrics is not None:
                axs[-1].text(bins[0], 0.2, f"10*EMD: {metrics[0]:.4f} $\pm$ {metrics[1]:.5f}"
                                           f"    ;    1e3*TriDist: {metrics[2]:.5f} $\pm$ "
                                           f"{metrics[3]:.4f}", fontsize=13)
                axs[-1].set_yticks([])
            unit = "" if observable.unit is None else f" [{observable.unit}]"
            axs[-1].set_xlabel(f"${{{observable.tex_label}}}${unit}")
            axs[-1].set_xscale(observable.xscale)
            axs[-1].set_xlim(bins[0], bins[-1])
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close()

    def hist_line(
        self,
        ax: mpl.axes.Axes,
        bins: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        label: str,
        color: str,
        linestyle: str = "solid",
        fill: bool = False,
    ):
        """
        Plot a stepped line for a histogram, optionally with error bars.
        Args:
            ax: Matplotlib Axes
            bins: Numpy array with bin boundaries
            y: Y values for the bins
            y_err: Y errors for the bins
            label: Label of the line
            color: Color of the line
            linestyle: line style
            fill: Filled histogram
        """

        dup_last = lambda a: np.append(a, a[-1])

        if fill:
            ax.fill_between(
                bins, dup_last(y), label=label, facecolor=color, step="post", alpha=0.2
            )
        else:
            ax.step(
                bins,
                dup_last(y),
                label=label,
                color=color,
                linewidth=1.0,
                where="post",
                ls=linestyle,
            )
        if y_err is not None:
            if len(y_err.shape) == 2:
                y_low = y_err[0]
                y_high = y_err[1]
            else:
                y_low = y - y_err
                y_high = y + y_err

            ax.step(
                bins,
                dup_last(y_high),
                color=color,
                alpha=0.5,
                linewidth=0.5,
                where="post",
            )
            ax.step(
                bins,
                dup_last(y_low),
                color=color,
                alpha=0.5,
                linewidth=0.5,
                where="post",
            )
            ax.fill_between(
                bins,
                dup_last(y_low),
                dup_last(y_high),
                facecolor=color,
                alpha=0.3,
                step="post",
            )

    def corner_text(
        self, ax: mpl.axes.Axes, text: str, horizontal_pos: str, vertical_pos: str
    ):
        ax.text(
            x=0.95 if horizontal_pos == "right" else 0.05,
            y=0.95 if vertical_pos == "top" else 0.05,
            s=text,
            horizontalalignment=horizontal_pos,
            verticalalignment=vertical_pos,
            transform=ax.transAxes,
        )
        # Dummy line for automatic legend placement
        plt.plot(
            0.8 if horizontal_pos == "right" else 0.2,
            0.8 if vertical_pos == "top" else 0.2,
            transform=ax.transAxes,
            color="none",
        )

    def plot_migration(self, file: str, gt_hard=False):
        if gt_hard:
            obs_hard = self.obs_hard
            name_hard = "Hard"
        else:
            obs_hard = self.obs_gen_single
            name_hard = "Unfold"
        cmap = plt.get_cmap('viridis')
        cmap.set_bad("white")
        with PdfPages(file) as pp:
            for obs, bins, data_reco, data_hard in zip(
                self.observables, self.bins, self.obs_reco, obs_hard
                    ):
                if self.bayesian and not gt_hard:
                    plt.hist2d(data_reco, data_hard[0], density=True, bins=bins, rasterized=True, cmap=cmap, norm=mpl.colors.LogNorm())
                else:
                    plt.hist2d(data_reco, data_hard, density=True, bins=bins, rasterized=True, cmap=cmap,
                               norm=mpl.colors.LogNorm())
                unit = "" if obs.unit is None else f" [{obs.unit}]"
                plt.title(f"${{{obs.tex_label}}}${unit}")
                plt.xlabel("Reco")
                plt.ylabel(name_hard)
                plt.xlim(bins[0], bins[-1])
                plt.ylim(bins[0], bins[-1])
                plt.colorbar()
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

    def plot_migration2(self, file: str, gt_hard=False, SB_hard=False):
        if gt_hard:
            obs_hard = self.obs_hard
            name_hard = "Hard"
        elif SB_hard:
            obs_hard = self.obs_compare
            name_hard = "SB"
        else:
            obs_hard = self.obs_gen_single
            name_hard = "Unfold"
        cmap = plt.get_cmap('viridis')
        cmap.set_bad("white")

        ranges = [
            [0, 0.4],
            [0, 0.3],
            [0, 0.5],
            [0, 0.5],
            [0, 0.2],
            [0, 0.25]
        ]

        with PdfPages(file) as pp:
            for obs, bins, data_reco, data_hard, r in zip(
                self.observables, self.bins, self.obs_reco, obs_hard, ranges
                    ):
                cmap = plt.get_cmap('viridis')
                cmap.set_bad("white")
                if gt_hard or SB_hard:
                    h, x, y = np.histogram2d(data_hard, data_reco, bins=(bins, bins))
                elif self.bayesian:
                    h, x, y = np.histogram2d(data_hard[0][0], data_reco, bins=(bins, bins))
                else:
                    h, x, y = np.histogram2d(data_hard[0], data_reco, bins=(bins, bins))
                h = np.ma.divide(h, np.sum(h, -1, keepdims=True)).filled(0)
                h[h == 0] = np.nan
                plt.pcolormesh(bins, bins, h, cmap=cmap, rasterized=True, vmin=r[0], vmax=r[1])
                plt.colorbar()

                unit = "" if obs.unit is None else f" [{obs.unit}]"
                plt.title(f"${{{obs.tex_label}}}${unit}")
                plt.xlim(bins[0], bins[-1])
                plt.ylim(bins[0], bins[-1])
                plt.xlabel("Reco")
                plt.ylabel(name_hard)

                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

    def compute_metrics(self):
        for i, obs in enumerate(self.observables):
            obs.metrics = {}
            x_gen = self.obs_gen_single[i]
            x_true = self.obs_hard[i]
            bins = self.bins[i]
            if not self.bayesian:
                emd_unfoldings = []
                triangle_unfoldings = []
                for n in range(self.n_unfoldings):
                    emd = GetEMD(x_true, x_gen[n], nboot=0)
                    triangle = get_triangle_distance(x_true, x_gen[n], bins, nboot=0)
                    emd_unfoldings.append(emd)
                    triangle_unfoldings.append(triangle)

                obs.metrics["emd_arr"] = [np.array(emd_unfoldings)]
                obs.metrics["triangle_arr"] = [np.array(triangle_unfoldings)]

                emd_full_mean, emd_full_std = GetEMD(x_true, x_gen.flatten(), nboot=3)
                triangle_full_mean, triangle_full_std = get_triangle_distance(x_true, x_gen.flatten(), bins, nboot=3)
                obs.metrics["emd_full"] = [round(emd_full_mean, 4)]
                obs.metrics["emd_full_std"] = [round(emd_full_std, 5)]
                obs.metrics["triangle_full"] = [round(triangle_full_mean, 4)]
                obs.metrics["triangle_full_std"] = [round(triangle_full_std, 5)]

            else:
                emd_arrs = []
                triangle_arrs = []
                emd_fulls = []
                emd_fulls_std = []
                triangle_fulls = []
                triangle_fulls_std = []
                for b in range(len(x_gen)):
                    x_gen_bayes = x_gen[b]
                    emd_unfoldings = []
                    triangle_unfoldings = []
                    for n in range(self.n_unfoldings):
                        emd = GetEMD(x_true, x_gen_bayes[n], nboot=0)
                        triangle = get_triangle_distance(x_true, x_gen_bayes[n], bins, nboot=0)
                        emd_unfoldings.append(emd)
                        triangle_unfoldings.append(triangle)

                    emd_arrs.append(np.array(emd_unfoldings))
                    triangle_arrs.append(np.array(triangle_unfoldings))

                    emd_full_mean, emd_full_std = GetEMD(x_true, x_gen_bayes.flatten(), nboot=3)
                    triangle_full_mean, triangle_full_std = get_triangle_distance(x_true, x_gen_bayes.flatten(), bins,
                                                                                  nboot=3)
                    emd_fulls.append(round(emd_full_mean, 4))
                    emd_fulls_std.append(round(emd_full_std, 5))
                    triangle_fulls.append(round(triangle_full_mean, 4))
                    triangle_fulls_std.append(round(triangle_full_std, 5))

                obs.metrics["emd_arr"] = np.stack(emd_arrs)
                obs.metrics["triangle_arr"] = np.stack(triangle_arrs)
                obs.metrics["emd_full"] = emd_fulls
                obs.metrics["emd_full_std"] = emd_fulls_std
                obs.metrics["triangle_full"] = triangle_fulls
                obs.metrics["triangle_full_std"] = triangle_fulls_std

                emd_full_full, emd_full_full_std = GetEMD(x_true, x_gen[1:].flatten(), nboot=3)
                triangle_full_full, triangle_full_full_std = get_triangle_distance(x_true, x_gen[1:].flatten(), bins,
                                                                              nboot=3)
                obs.metrics["emd_full_full"] = emd_full_full
                obs.metrics["emd_full_full_std"] = emd_full_full_std
                obs.metrics["triangle_full_full"] = triangle_full_full
                obs.metrics["triangle_full_full_std"] = triangle_full_full_std

    def hist_metrics_unfoldings(self, file: str, pickle_file: Optional[str] = None):
        pickle_data = {'emd': [], 'triangle': []}
        with PdfPages(file) as pp:
            for obs in self.observables:
                nbins = 64
                emd_bins = np.linspace(0, 1.5 * max(obs.metrics["emd_arr"][0]), nbins)
                triangle_bins = np.linspace(0, 1.5 * max(obs.metrics["triangle_arr"][0]), nbins)
                emd_hist, _ = np.histogram(obs.metrics["emd_arr"][0], bins=emd_bins, density=False)
                triangle_hist, _ = np.histogram(obs.metrics["triangle_arr"][0], bins=triangle_bins, density=False)

                emd_lines = [
                    Line(
                        y=emd_hist,
                        y_err=None,
                        label="EMD Unfoldings",
                        color=self.colors[0],
                    ),
                    Line(
                        y=obs.metrics["emd_full"][0],
                        vline=True,
                        y_err=None,
                        label="EMD Full",
                        color=self.colors[1],
                        linestyle='dashed',
                    )
                ]
                triangle_lines = [
                    Line(
                        y=triangle_hist,
                        y_err=None,
                        label="Triangle Unfoldings",
                        color=self.colors[0],
                    ),
                    Line(
                        y=obs.metrics["triangle_full"][0],
                        vline=True,
                        y_err=None,
                        label="Triangle Full",
                        color=self.colors[1],
                        linestyle='dashed',
                    )
                ]

                if self.bayesian:
                    emd_lines.append(Line(
                        y=obs.metrics["emd_full_full"],
                        vline=True,
                        y_err=None,
                        label="EMD FullFull",
                        color=self.colors[2],
                        linestyle='dashed',
                    ))
                    triangle_lines.append(Line(
                        y=obs.metrics["triangle_full_full"],
                        vline=True,
                        y_err=None,
                        label="Triangle FullFull",
                        color=self.colors[2],
                        linestyle='dashed',
                    ))

                self.hist_plot(pp, emd_lines, emd_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')
                self.hist_plot(pp, triangle_lines, triangle_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')

                if pickle_file is not None:
                    pickle_data["emd"].append(obs.metrics["emd_arr"])
                    pickle_data["triangle"].append(obs.metrics["triangle_arr"])
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def hist_metrics_bayesian(self, file: str, pickle_file: Optional[str] = None):
        pickle_data = {'emd': [], 'triangle': []}

        with PdfPages(file) as pp:
            for obs in self.observables:
                nbins = 64
                emd_bins = np.linspace(0, 1.5 * max(obs.metrics["emd_arr"][:, 0]), nbins)
                triangle_bins = np.linspace(0, 1.5 * max(obs.metrics["triangle_arr"][:, 0]), nbins)
                emd_hist, _ = np.histogram(obs.metrics["emd_arr"][:, 0], bins=emd_bins, density=False)
                triangle_hist, _ = np.histogram(obs.metrics["triangle_arr"][:, 0], bins=triangle_bins, density=False)

                emd_lines = [
                    Line(
                        y=emd_hist,
                        y_err=None,
                        label="EMD Bayesians",
                        color=self.colors[0],
                    ),
                    Line(
                        y=obs.metrics["emd_full_full"],
                        vline=True,
                        y_err=None,
                        label="EMD FullFull",
                        color=self.colors[1],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["emd_arr"][0][0],
                        vline=True,
                        y_err=None,
                        label="EMD MAP",
                        color=self.colors[2],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["emd_full"][0],
                        vline=True,
                        y_err=None,
                        label="EMD MAP FullUn",
                        color=self.colors[3],
                        linestyle='dashed',
                    )
                ]
                triangle_lines = [
                    Line(
                        y=triangle_hist,
                        y_err=None,
                        label="Triangle Bayesians",
                        color=self.colors[0],
                    ),
                    Line(
                        y=obs.metrics["triangle_full_full"],
                        vline=True,
                        y_err=None,
                        label="Triangle FullFull",
                        color=self.colors[1],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["triangle_arr"][0][0],
                        vline=True,
                        y_err=None,
                        label="Triangle MAP",
                        color=self.colors[2],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["triangle_full"][0],
                        vline=True,
                        y_err=None,
                        label="Triangle MAP FullUn",
                        color=self.colors[3],
                        linestyle='dashed',
                    )
                ]

                self.hist_plot(pp, emd_lines, emd_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')
                self.hist_plot(pp, triangle_lines, triangle_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')

                if pickle_file is not None:
                    pickle_data["emd"].append(obs.metrics["emd_arr"])
                    pickle_data["triangle"].append(obs.metrics["triangle_arr"])
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_multiple_unfoldings(self, file: str):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare
            ):
                if self.bayesian:
                    data = data_gen[0]
                else:
                    data = data_gen
                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data.flatten(), bayesian=False)
                lines = [
                    Line(
                        y=y_hard,
                        y_err=None,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=None,
                        label="Full",
                        color=self.colors[0],
                    ),
                ]

                hists = np.stack(
                    [np.histogram(d, bins=bins, density=True)[0] for d in data], axis=0
                )
                y_gen_mean = np.mean(hists, axis=0)

                lines.append(
                    Line(
                        y=y_gen_mean,
                        y_err=None,
                        y_ref=y_hard,
                        label=f"Mean",
                        color=self.colors[2]
                    )
                )

                obs_emds = obs.metrics["emd_arr"][0]
                emd_argsort = np.argsort(obs_emds)

                for i in range(1):
                    index = emd_argsort[i]
                    d = data[index]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"Best, {i+1}",
                            color=self.colors[4+i]
                        )
                    )
                    index = emd_argsort[-(i+1)]
                    d = data[index]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"Worst, {i+1}",
                            color=self.colors[7 + i]
                        )
                    )
                self.hist_plot(pp, lines, bins, obs, metrics=None)

    def plot_multiple_bayesians(self, file: str):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare
            ):
                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data_gen[:, 0], bayesian=self.bayesian)
                lines = [
                    Line(
                        y=y_hard,
                        y_err=None,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=None,
                        y_ref=y_hard,
                        label="MAP",
                        color=self.colors[1],
                    ),
                ]

                hists = np.stack(
                    [np.histogram(d, bins=bins, density=True)[0] for d in data_gen[1:, 0]], axis=0
                )
                y_gen_mean = np.mean(hists, axis=0)

                lines.append(
                    Line(
                        y=y_gen_mean,
                        y_err=None,
                        y_ref=y_hard,
                        label=f"Mean",
                        color=self.colors[2]
                    )
                )

                obs_emds = obs.metrics["emd_arr"][:, 0]
                emd_argsort = np.argsort(obs_emds)

                for i in range(1):
                    index = emd_argsort[i]
                    d = data_gen[index, 0]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"Best, {index}",
                            color=self.colors[4+i]
                        )
                    )
                    index = emd_argsort[-(i+1)]
                    d = data_gen[index]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"Worst, {index}",
                            color=self.colors[7 + i]
                        )
                    )
                self.hist_plot(pp, lines, bins, obs, metrics=None)

    def plot_observables_full(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare
            ):
                if self.bayesian:
                    bay_n, dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(bay_n, -1)
                else:
                    dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(-1)


                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_reco, y_reco_err = self.compute_hist_data(bins, data_reco, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data, bayesian=self.bayesian)
                lines = [
                    Line(
                        y=y_reco,
                        y_err=y_reco_err,
                        label="Reco",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label=f"MAP {dist_n} Unf. full",
                        color=self.colors[1],
                    ),

                ]
                if self.bayesian:
                    y_gen_full, _ = self.compute_hist_data(bins, data.reshape(-1), bayesian=False)
                    lines.append(Line(
                        y=y_gen_full,
                        y_err=None,
                        y_ref=y_hard,
                        label=f"{bay_n} bay., {dist_n} Unf.",
                        color=self.colors[3],
                    ))
                if self.compare:
                    y_comp, y_comp_err = self.compute_hist_data(bins, data_compare, bayesian=False)
                    lines.append(
                        Line(
                            y=y_comp,
                            y_err=y_gen_err,
                            y_ref=y_hard,
                            label="SB",
                            color=self.colors[3]
                        )
                    )
                if not self.bayesian:
                    metrics = [obs.metrics["emd_full"][0],
                               obs.metrics["emd_full_std"][0],
                               obs.metrics["triangle_full"][0],
                               obs.metrics["triangle_full_std"][0]]
                else:
                    metrics = [obs.metrics["emd_full_full"],
                               obs.metrics["emd_full_full_std"],
                               obs.metrics["triangle_full_full"],
                               obs.metrics["triangle_full_full_std"]]
                self.hist_plot(pp, lines, bins, obs, metrics=metrics)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def save_metrics(self, pickle_file):

        with open(pickle_file, "wb") as f:
            pickle.dump(self.observables, f)


class OmnifoldPlots(Plots):
    def __init__(
        self,
        observables: list[Observable],
        losses: dict,
        x_hard: torch.Tensor,
        x_reco: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        bayesian: bool = False,
        show_metrics: bool = True
    ):
        """
        Initializes the plotting pipeline with the data to be plotted.
        Args:
            doc: Documenter object
            observables: List of observables
            losses: Dictionary with loss terms and learning rate as a function of the epoch
            x_hard: Hard level data
            x_reco: Reco level data
            labels: True labels
            predictions: Predicted labels
        """
        self.observables = observables
        self.losses = losses
        self.labels = labels.cpu().bool().numpy()
        self.predictions = predictions.cpu().numpy()
        self.weights = np.clip((1. - self.predictions)/self.predictions, 0, 10).squeeze()

        self.obs_hard = []
        self.obs_reco = []
        self.bins = []
        for obs in observables:
            o_hard = obs.compute(x_hard)
            o_reco = obs.compute(x_reco)
            self.obs_hard.append(o_hard.cpu().numpy())
            self.obs_reco.append(o_reco.cpu().numpy())
            self.bins.append(obs.bins(o_hard).cpu().numpy())

        self.bayesian = bayesian
        self.show_metrics = show_metrics
        if self.show_metrics:
            print(f"    Computing metrics")
            self.compute_metrics()

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        self.colors = [f"C{i}" for i in range(10)]

    def compute_hist_data(self, bins: np.ndarray, data: np.ndarray, bayesian=False, weights=None):
        if bayesian:
            hists = np.stack(
                [np.histogram(data, bins=bins, density=True, weights=weight_sample)[0] for weight_sample in weights], axis=0
            )
            #y = np.median(hists, axis=0)
            y = hists[0]
            #y_err = np.stack(
            #    (np.quantile(hists, 0.159, axis=0), np.quantile(hists, 0.841, axis=0)),
            #    axis=0,
            #)
            y_err = np.std(hists, axis=0)
        else:
            y, _ = np.histogram(data, bins=bins, density=False, weights=weights)
            y_err = np.sqrt(y)
        return y, y_err

    def plot_classes(self, file: str):
        """
        Makes plots of truth and predicted classes for all observables.
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, bins, data in zip(self.observables, self.bins, self.obs_reco):
                binned_classes_test, _, _ = binned_statistic(data, self.labels.T, bins=bins)
                if self.bayesian:
                    binned_classes_predict, _, _ = binned_statistic(data, self.predictions[0].T, bins=bins)
                else:
                    binned_classes_predict, _, _ = binned_statistic(data, self.predictions.T, bins=bins)
                bin_totals, _ = np.histogram(data, bins=bins)
                y_true = np.stack(binned_classes_test, axis=1)
                y_predict = np.stack(binned_classes_predict, axis=1)
                y_true_err = np.sqrt(y_true * (1 - y_true) / bin_totals[:, None])

                lines = []
                for i in range(y_true.shape[-1]):
                    lines.append(Line(
                        y=y_true[:,i],
                        y_err=y_true_err[:,i],
                        color=self.colors[i],
                        linestyle="dashed",
                    ))
                    lines.append(Line(
                        y=y_predict[:,i],
                        y_ref=y_true[:,i],
                        label=f"{i} extra" if y_true.shape[-1] > 1 else "acceptance",
                        color=self.colors[i],
                    ))
                self.hist_plot(pp, lines, bins, obs, no_scale=True, yscale="linear")

    def plot_reco(self, file: str):
        """
        Makes plots of truth and predicted classes for all observables.
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, bins, data in zip(self.observables, self.bins, self.obs_reco):

                data_pythia = data[self.labels.squeeze()]
                data_herwig = data[~self.labels.squeeze()]
                weights_pythia = self.weights[..., self.labels.squeeze()]

                y_pythia, y_pythia_err = self.compute_hist_data(bins, data_pythia, bayesian=False)
                y_herwig, y_herwig_err = self.compute_hist_data(bins, data_herwig, bayesian=False)
                y_reweight, y_reweight_err = self.compute_hist_data(bins, data_pythia, bayesian=self.bayesian, weights=weights_pythia)

                lines = [
                    Line(
                        y=y_pythia,
                        y_err=y_pythia_err,
                        label="Pythia Reco",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_herwig,
                        y_err=y_herwig_err,
                        label="Herwig Reco",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_reweight,
                        y_err=y_reweight_err,
                        y_ref=y_herwig,
                        label="Pyth. Rew.",
                        color=self.colors[1],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs, metrics=None)

    def plot_hard(self, file: str):
        """
        Makes plots of truth and predicted classes for all observables.
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, bins, data in zip(self.observables, self.bins, self.obs_hard):
                data_pythia = data[self.labels.squeeze()]
                data_herwig = data[~self.labels.squeeze()]
                weights_pythia = self.weights[..., self.labels.squeeze()]

                y_pythia, y_pythia_err = self.compute_hist_data(bins, data_pythia, bayesian=False)
                y_herwig, y_herwig_err = self.compute_hist_data(bins, data_herwig, bayesian=False)
                y_reweight, y_reweight_err = self.compute_hist_data(bins, data_pythia, bayesian=self.bayesian,
                                                                    weights=weights_pythia)

                lines = [
                    Line(
                        y=y_pythia,
                        y_err=y_pythia_err,
                        label="Pythia Hard",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_herwig,
                        y_err=y_herwig_err,
                        label="Herwig Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_reweight,
                        y_err=y_reweight_err,
                        y_ref=y_herwig,
                        label="Pyth. Rew.",
                        color=self.colors[1],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs, metrics=None)

    def plot_observables(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco in zip(
            self.observables, self.bins, self.obs_hard, self.obs_reco):

                data_reco = data_reco[~self.labels.squeeze()]
                data_hard_herwig = data_hard[~self.labels.squeeze()]
                data_hard_pythia = data_hard[self.labels.squeeze()]
                weights = self.weights[..., self.labels.squeeze()]

                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard_herwig, bayesian=False)
                y_reco, y_reco_err = self.compute_hist_data(bins, data_reco, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data_hard_pythia, bayesian=self.bayesian, weights=weights)
                lines = [
                    Line(
                        y=y_reco,
                        y_err=y_reco_err,
                        label="Reco",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label="Unfold",
                        color=self.colors[1],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs, metrics=None)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def compute_metrics(self):
        for obs, bins, data_hard, data_reco in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco):

            data_hard_herwig = data_hard[~self.labels.squeeze()]
            data_hard_pythia = data_hard[self.labels.squeeze()]
            weights = self.weights[..., self.labels.squeeze()]

            if not self.bayesian:
                emd_mean, emd_std = GetEMD(data_hard_herwig, data_hard_pythia, nboot=10, weights_arr=weights)
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(data_hard_herwig, data_hard_pythia, bins, nboot=10, weights=weights)
            else:
                emd_mean, emd_std = GetEMD(data_hard_herwig, data_hard_pythia, nboot=10, weights_arr=weights[0])
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(data_hard_herwig, data_hard_pythia, bins,
                                                                              nboot=10, weights=weights[0])
            #else:
            #    emd = []
            #    triangle_dist = []
            #    for weight_sample in weights:
            #        emd_sample, _ = GetEMD(data_hard_herwig, data_hard_pythia, nboot=1, weights_arr=weight_sample)
            #        triangle_dist_sample, _ = get_triangle_distance(data_hard_herwig, data_hard_pythia, bins, nboot=1, weights=weight_sample)
            #        emd.append(emd_sample)
            #        triangle_dist.append(triangle_dist_sample)
            #    emd_mean = np.array(emd).mean()
            #    emd_std = np.array(emd).std()
            #    triangle_dist_mean = np.array(triangle_dist).mean()
            #    triangle_dist_std = np.array(triangle_dist).std()

            obs.emd_mean = round(emd_mean, 4)
            obs.emd_std = round(emd_std, 5)
            obs.triangle_mean = round(triangle_dist_mean, 4)
            obs.triangle_std = round(triangle_dist_std, 5)


class TTBar_Plots(Plots):
    """
    Implements the plotting pipeline to evaluate the performance of conditional generative
    networks.
    """

    def __init__(
        self,
        observables: list[Observable],
        losses: dict,
        x_hard: torch.Tensor,
        x_gen_single: torch.Tensor,
        x_gen_dist: torch.Tensor=None,
        x_hard_pp=None,
        x_reco_pp=None,
        bayesian: bool = False
    ):
        """
        Initializes the plotting pipeline with the data to be plotted.
        Args:
            doc: Documenter object
            observables: List of observables
            losses: Dictionary with loss terms and learning rate as a function of the epoch
            x_test: Data from test dataset
            x_gen_single: Generated data (single point for each test sample)
            x_gen_dist: Generated data (multiple points for subset of test data)
            event_type: Event types for the test dataset
            bayesian: Boolean that indicates if the model is bayesian
        """
        self.observables = observables
        self.losses = losses

        self.x_hard_pp = x_hard_pp
        self.x_reco_pp = x_reco_pp

        self.obs_hard = []
        self.obs_gen_single = []
        self.obs_gen_dist = []
        self.bins = []

        for obs in observables:
            o_hard = obs.compute(x_hard)
            self.obs_hard.append(o_hard.cpu().numpy())

            o_gen_single = obs.compute(x_gen_single)
            self.obs_gen_single.append(o_gen_single.cpu().numpy())

            o_gen_dist = obs.compute(x_gen_dist)
            self.obs_gen_dist.append(o_gen_dist.cpu().numpy())

            self.bins.append(obs.bins(o_hard).cpu().numpy())

        self.n_unfoldings = x_gen_single.shape[-3]
        self.bayesian = bayesian
        self.colors = [f"C{i}" for i in range(10)]

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)


    def plot_losses(self, file: str):
        """
        Makes plots of the losses (total loss and if bayesian, BCE loss and KL loss
        separately) and learning rate as a function of the epoch.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            self.plot_single_loss(
                pp,
                "loss",
                (self.losses["tr_loss"], self.losses["val_loss"]),
                ("train", "val"),
            )
            if self.bayesian:
                self.plot_single_loss(
                    pp,
                    "INN loss",
                    (self.losses["train_inn_loss"], self.losses["val_inn_loss"]),
                    ("train", "val"),
                )
                self.plot_single_loss(
                    pp,
                    "KL loss",
                    (self.losses["train_kl_loss"], self.losses["val_kl_loss"]),
                    ("train", "val"),
                )
            self.plot_single_loss(
                pp, "learning rate", (self.losses["lr"],), (None,), "log"
            )

    def plot_single_loss(
        self,
        pp: PdfPages,
        ylabel: str,
        curves: tuple[np.ndarray],
        labels: tuple[str],
        yscale: str = "linear",
    ):
        """
        Makes single loss plot.
        Args:
            pp: Multipage PDF object
            ylabel: Y axis label
            curves: List of numpy arrays with the loss curves to be plotted
            labels: Labels of the loss curves
            yscale: Y axis scale, "linear" or "log"
        """
        fig, ax = plt.subplots(figsize=(4, 3.5))
        for i, (curve, label) in enumerate(zip(curves, labels)):
            epochs = np.arange(6, len(curve) + 6)
            ax.plot(epochs[5:], curve[5:], label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        if any(label is not None for label in labels):
            ax.legend(loc="center right", frameon=False)
        plt.savefig(pp, format="pdf", bbox_inches="tight")
        plt.close()

    def plot_observables(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_gen in zip(
                self.observables, self.bins, self.obs_hard, self.obs_gen_single
            ):
                if self.bayesian:
                    data = data_gen[:, 0]
                else:
                    data = data_gen[0]

                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data, bayesian=self.bayesian)
                lines = [
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label="Unfold",
                        color=self.colors[1],
                    ),
                ]
                if self.bayesian:
                    bay_n, dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(bay_n, -1)
                else:
                    dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(-1)
                y_gen_full, y_gen_err = self.compute_hist_data(bins, data, bayesian=self.bayesian)

                lines.append(
                    Line(
                        y=y_gen_full,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label=f"{dist_n} Unfoldings",
                        color=self.colors[3]
                    )
                )
                self.hist_plot(pp, lines, bins, obs, metrics=None)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_preprocessed(self, file: str):
        """
        Plots preprocessed observable distributions
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for i, data_hard_pp in enumerate(self.x_hard_pp.T):
                try:
                    bins = 100
                    plt.hist(data_hard_pp.cpu().numpy(), bins=bins, density=True)
                    plt.title(f"Hard PP, dim {i}")
                    plt.savefig(pp, format="pdf", bbox_inches="tight")
                    plt.close()
                except:
                    print(f" dim {i} of hard failed")


            for i, data_hard_pp in enumerate(self.x_reco_pp.T):
                bins = 100
                plt.hist(data_hard_pp.cpu().numpy(), bins=bins, density=True)
                plt.title(f"Reco PP, dim {i}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()
