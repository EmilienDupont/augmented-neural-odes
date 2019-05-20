import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


categorical_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

all_categorical_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def vector_field_plt(odefunc, num_points, timesteps, inputs=None, targets=None,
                     model=None, h_min=-2., h_max=2., t_max=1., extra_traj=[],
                     save_fig=''):
    """For a 1 dimensional odefunc, returns the vector field associated with the
    function.

    Parameters
    ----------
    odefunc : ODEfunc instance
        Must be 1 dimensional ODE. I.e. dh/dt = f(h, t) with h being a scalar.

    num_points : int
        Number of points in h at which to evaluate f(h, t).

    timesteps : int
        Number of timesteps at which to evaluate f(h, t).

    inputs : torch.Tensor or None
        Shape (num_points, 1). Input points to ODE.

    targets : torch.Tensor or None
        Shape (num_points, 1). Target points for ODE.

    model : anode.models.ODEBlock instance or None
        If model is passed as argument along with inputs, it will be used to
        compute the trajectory of each point in inputs and will be overlayed on
        the plot.

    h_min : float
        Minimum value of hidden state h.

    h_max : float
        Maximum value of hidden state h.

    t_max : float
        Maximum time to which we solve ODE.

    extra_traj : list of tuples
        Each tuple contains a list of numbers corresponding to the trajectory
        and a string defining the color of the trajectory. These will be dotted.
    """
    t, hidden, dtdt, dhdt = ode_grid(odefunc, num_points, timesteps,
                                     h_min=h_min, h_max=h_max, t_max=t_max)
    # Create meshgrid and vector field plot
    t_grid, h_grid = np.meshgrid(t, hidden, indexing='ij')
    plt.quiver(t_grid, h_grid, dtdt, dhdt, width=0.004, alpha=0.6)

    # Optionally add input points
    if inputs is not None:
        if targets is not None:
            color = ['red' if targets[i, 0] > 0 else 'blue' for i in range(len(targets))]
        else:
            color = 'red'
        # Input points are defined at t=0, i.e. at x=0 on the plot
        plt.scatter(x=[0] * len(inputs), y=inputs[:, 0].numpy(), c=color, s=80)

    # Optionally add target points
    if targets is not None:
        color = ['red' if targets[i, 0] > 0 else 'blue' for i in range(len(targets))]
        # Target points are defined at t=1, i.e. at x=1 on the plot
        plt.scatter(x=[t_max] * len(targets), y=targets[:, 0].numpy(), c=color,
                    s=80)

    if model is not None and inputs is not None:
        color = ['red' if targets[i, 0] > 0 else 'blue' for i in range(len(targets))]
        for i in range(len(inputs)):
            init_point = inputs[i:i+1]
            trajectory = model.trajectory(init_point, timesteps)
            plt.plot(t, trajectory[:, 0, 0].detach().numpy(), c=color[i],
                     linewidth=2)

    if len(extra_traj):
        for traj, color in extra_traj:
            num_steps = len(traj)
            t_traj = [t_max * float(i) / (num_steps - 1) for i in range(num_steps)]
            plt.plot(t_traj, traj, c=color, linestyle='--', linewidth=2)
            plt.scatter(x=t_traj[1:], y=traj[1:], c=color, s=20)

    plt.xlabel("t")
    plt.ylabel("h(t)")

    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=400, bbox_inches='tight')
        plt.clf()
        plt.close()


def histories_plt(all_history_info, plot_type='loss', shaded_err=False,
                  labels=[], include_mean=True, nfe_type='nfe',
                  time_per_epoch=[], save_fig=''):
    """
    Parameters
    ----------
    all_history_info : list
        results[i]["models"] coming out of experiment

    plot_type : string
        One of 'loss', 'nfe' or 'nfe_vs_loss'.

    shaded_err : bool
        If True, plots the standard deviation of the history as a shaded area
        around the mean.

    labels : list of string
        If len(labels) > 0, will color and annotate plot by desciprition in
        labels.

    include_mean : bool
        If False doesn't include mean of histories on the plot. This is useful
        when having incomplete histories (e.g. when a model underflows).

    nfe_type : string
        Only used when doing either an 'nfe' or 'nfe_vs_loss' plot.

    time_per_epoch : list of floats
        If empty, plots number of epochs on the x-axis. If not empty, scales
        the length of the x-axis by time per epoch for each model. The time per
        epoch should be given in seconds.

    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    for i, history_info in enumerate(all_history_info):
        model_type = history_info["type"]
        if len(labels) > 0:
            color = categorical_colors[i % 4]
            label = labels[i]
        else:
            if model_type == 'resnet':
                color = categorical_colors[0]
                label = 'ResNet'
            if model_type == 'odenet':
                color = categorical_colors[1]
                label = 'Neural ODE'
            if model_type == 'anode':
                color = categorical_colors[2]
                label = 'ANODE'

        # No concept of number of function evaluations for ResNet
        if model_type == 'resnet' and plot_type != 'loss':
            continue

        if plot_type == 'loss':
            histories = history_info["epoch_loss_history"]
            ylabel = "Loss"
        elif plot_type == 'nfe':
            if nfe_type == 'nfe':
                histories = history_info["epoch_nfe_history"]
            elif nfe_type == 'bnfe':
                histories = history_info["epoch_bnfe_history"]
            elif nfe_type == 'total_nfe':
                histories = history_info["epoch_total_nfe_history"]
            ylabel = "# of Function Evaluations"
        elif plot_type == 'nfe_vs_loss':
            histories_loss = history_info["epoch_loss_history"]
            if nfe_type == 'nfe':
                histories_nfe = history_info["epoch_nfe_history"]
            elif nfe_type == 'bnfe':
                histories_nfe = history_info["epoch_bnfe_history"]
            elif nfe_type == 'total_nfe':
                histories_nfe = history_info["epoch_total_nfe_history"]
            xlabel = "# of Function Evaluations"
            ylabel = "Loss"

        if plot_type == 'loss' or plot_type == 'nfe':
            if len(time_per_epoch):
                xlabel = "Time (seconds)"
            else:
                xlabel = "Epochs"

            if include_mean:
                mean_history = np.array(histories).mean(axis=0)
                if len(time_per_epoch):
                    epochs = time_per_epoch[i] * np.arange(len(histories[0]))
                else:
                    epochs = list(range(len(histories[0])))

                if shaded_err:
                    std_history = np.array(histories).std(axis=0)
                    plt.fill_between(epochs, mean_history - std_history,
                                     mean_history + std_history, facecolor=color,
                                     alpha=0.5)
                else:
                    for history in histories:
                        plt.plot(epochs, history, c=color, alpha=0.1)

                plt.plot(epochs, mean_history, c=color, label=label)
            else:
                for history in histories:
                    plt.plot(history, c=color, alpha=0.1)
        else:
            for j in range(len(histories_loss)):
                if j == 0:  # This is hacky, only used to avoid repeated labels
                    plt.scatter(histories_nfe[j], histories_loss[j], c=color,
                                alpha=0.5, label=label, linewidths=0)
                else:
                    plt.scatter(histories_nfe[j], histories_loss[j], c=color,
                                alpha=0.5, linewidths=0)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)

    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=400, bbox_inches='tight')
        plt.clf()
        plt.close()


def single_feature_plt(features, targets, save_fig=''):
    """Plots a feature map with points colored by their target value. Works for
    2 or 3 dimensions.

    Parameters
    ----------
    features : torch.Tensor
        Tensor of shape (num_points, 2) or (num_points, 3).

    targets : torch.Tensor
        Target points for ODE. Shape (num_points, 1). -1 corresponds to blue
        while +1 corresponds to red.

    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    alpha = 0.5
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    num_dims = features.shape[1]

    if num_dims == 2:
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=0)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)
        ax = plt.gca()
    elif num_dims == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(),
                   features[:, 2].numpy(), c=color, alpha=alpha,
                   linewidths=0, s=80)
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, right=False, left=False,
                       labelleft=False)

    ax.set_aspect(get_square_aspect_ratio(ax))

    if len(save_fig):
        fig.savefig(save_fig, format='png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def multi_feature_plt(features, targets, save_fig=''):
    """Plots multiple feature maps colored by their target value. Works for 2 or
    3 dimensions.

    Parameters
    ----------
    features : list of torch.Tensor
        Each list item has shape (num_points, 2) or (num_points, 3).

    targets : torch.Tensor
        Target points for ODE. Shape (num_points, 1). -1 corresponds to blue
        while +1 corresponds to red.

    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    alpha = 0.5
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    num_dims = features[0].shape[1]

    if num_dims == 2:
        fig, axarr = plt.subplots(1, len(features), figsize=(20, 10))
        for i in range(len(features)):
            axarr[i].scatter(features[i][:, 0].numpy(), features[i][:, 1].numpy(),
                             c=color, alpha=alpha, linewidths=0)
            axarr[i].tick_params(axis='both', which='both', bottom=False,
                                 top=False, labelbottom=False, right=False,
                                 left=False, labelleft=False)
            axarr[i].set_aspect(get_square_aspect_ratio(axarr[i]))
    elif num_dims == 3:
        fig = plt.figure(figsize=(20, 10))
        for i in range(len(features)):
            ax = fig.add_subplot(1, len(features), i + 1, projection='3d')

            ax.scatter(features[i][:, 0].numpy(), features[i][:, 1].numpy(),
                       features[i][:, 2].numpy(), c=color, alpha=alpha,
                       linewidths=0, s=80)
            ax.tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, right=False, left=False,
                           labelleft=False)
            ax.set_aspect(get_square_aspect_ratio(ax))

    fig.subplots_adjust(wspace=0.01)

    if len(save_fig):
        fig.savefig(save_fig, format='png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def trajectory_plt(model, inputs, targets, timesteps, highlight_inputs=False,
                   include_arrow=False, save_fig=''):
    """Plots trajectory of input points when evolved through model. Works for 2
    and 3 dimensions.

    Parameters
    ----------
    model : anode.models.ODENet instance

    inputs : torch.Tensor
        Shape (num_points, num_dims) where num_dims = 1, 2 or 3 depending on
        augment_dim.

    targets : torch.Tensor
        Shape (num_points, 1).

    timesteps : int
        Number of timesteps to calculate for trajectories.

    highlight_inputs : bool
        If True highlights input points by drawing edge around points.

    include_arrow : bool
        If True adds an arrow to indicate direction of trajectory.

    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    alpha = 0.5
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    # Calculate trajectories (timesteps, batch_size, input_dim)
    trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    # Features are trajectories at the final time
    features = trajectories[-1]

    if model.augment_dim > 0:
        aug = torch.zeros(inputs.shape[0], model.odeblock.odefunc.augment_dim)
        inputs_aug = torch.cat([inputs, aug], 1)
    else:
        inputs_aug = inputs

    input_dim = model.data_dim + model.augment_dim

    if input_dim == 2:
        # Plot starting and ending points of trajectories
        input_linewidths = 2 if highlight_inputs else 0
        plt.scatter(inputs_aug[:, 0].numpy(), inputs_aug[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=input_linewidths, edgecolor='orange')
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=0)

        # For each point in batch, plot its trajectory
        for i in range(inputs_aug.shape[0]):
            # Plot trajectory
            trajectory = trajectories[:, i, :]
            x_traj = trajectory[:, 0].numpy()
            y_traj = trajectory[:, 1].numpy()
            plt.plot(x_traj, y_traj, c=color[i], alpha=alpha)
            # Optionally add arrow to indicate direction of flow
            if include_arrow:
                arrow_start = x_traj[-2], y_traj[-2]
                arrow_end = x_traj[-1], y_traj[-1]
                plt.arrow(arrow_start[0], arrow_start[1],
                          arrow_end[0] - arrow_start[0],
                          arrow_end[1] - arrow_start[1], shape='full', lw=0,
                          length_includes_head=True, head_width=0.15,
                          color=color[i], alpha=alpha)

        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

        ax = plt.gca()
    elif input_dim == 3:
        # Create figure
        fig = plt.figure()
        ax = Axes3D(fig)

        # Plot starting and ending points of trajectories
        input_linewidths = 1 if highlight_inputs else 0
        ax.scatter(inputs_aug[:, 0].numpy(), inputs_aug[:, 1].numpy(),
                   inputs_aug[:, 2].numpy(), c=color, alpha=alpha,
                   linewidths=input_linewidths, edgecolor='orange')
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(),
                   features[:, 2].numpy(), c=color, alpha=alpha, linewidths=0)

        # For each point in batch, plot its trajectory
        for i in range(inputs_aug.shape[0]):
            # Plot trajectory
            trajectory = trajectories[:, i, :]
            x_traj = trajectory[:, 0].numpy()
            y_traj = trajectory[:, 1].numpy()
            z_traj = trajectory[:, 2].numpy()
            ax.plot(x_traj, y_traj, z_traj, c=color[i], alpha=alpha)

            # Optionally add arrow
            if include_arrow:
                arrow_start = x_traj[-2], y_traj[-2], z_traj[-2]
                arrow_end = x_traj[-1], y_traj[-1], z_traj[-1]

                arrow = Arrow3D([arrow_start[0], arrow_end[0]],
                                [arrow_start[1], arrow_end[1]],
                                [arrow_start[2], arrow_end[2]],
                                mutation_scale=15,
                                lw=0, color=color[i], alpha=alpha)
                ax.add_artist(arrow)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        raise RuntimeError("Input dimension must be 2 or 3 but was {}".format(input_dim))

    ax.set_aspect(get_square_aspect_ratio(ax))

    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=400, bbox_inches='tight')
        plt.clf()
        plt.close()


def input_space_plt(model, plot_range=(-2., 2.), num_steps=201, save_fig=''):
    """Plots input space, where each grid point is colored by the value
    predicted by the model at that point. This only works for 2 dimensional
    inputs.

    Parameters
    ----------
    model : anode.models.ODENet

    plot_range : tuple of floats
        Range on which to plot input space.

    num_steps : int
        Number of steps at which to evalute model along each dimension.

    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    # Grid the input space
    grid = torch.zeros((num_steps * num_steps, 2))
    idx = 0
    for x1 in np.linspace(plot_range[0], plot_range[1], num_steps):
        for x2 in np.linspace(plot_range[0], plot_range[1], num_steps):
            grid[idx, :] = torch.Tensor([x1, x2])
            idx += 1

    # Calculate values predicted by model on grid
    predictions = model(grid)
    pred_grid = predictions.view(num_steps, num_steps).detach()

    # Set up a custom color map where -1 is mapped to blue and 1 to red
    colors = [(1, 1, 1), (0, 0, 1), (0.5, 0, 0.5), (1, 0, 0), (1, 1, 1)]
    colormap = LinearSegmentedColormap.from_list('cmap_red_blue', colors, N=300)

    # Plot input space as a heatmap
    plt.imshow(pred_grid, vmin=-2., vmax=2., cmap=colormap, alpha=0.75)
    plt.colorbar()
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=400, bbox_inches='tight')
        plt.clf()
        plt.close()

# Helper functions and classes

class Arrow3D(FancyArrowPatch):
    """Class used to draw arrows on 3D plots. Taken from:
    https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def ode_grid(odefunc, num_points, timesteps, h_min=-2., h_max=2., t_max=1.):
    """For a 1 dimensional odefunc, returns the points and derivatives at every
    point on a grid. This is useful for plotting vector fields.

    Parameters
    ----------
    odefunc : anode.models.ODEfunc instance
        Must be 1 dimensional ODE. I.e. dh/dt = f(h, t) with h being a scalar.

    num_points : int
        Number of points in h at which to evaluate f(h, t).

    timesteps : int
        Number of timesteps at which to evaluate f(h, t).

    h_min : float
        Minimum value of hidden state h.

    h_max : float
        Maximum value of hidden state h.

    t_max : float
        Maximum time for ODE solution.
    """
    # Vector field is defined at every point (t[i], hidden[j])
    t = np.linspace(0., t_max, timesteps)
    hidden = np.linspace(h_min, h_max, num_points)
    # Vector at each point in vector field is (dt/dt, dh/dt)
    dtdt = np.ones((timesteps, num_points))  # dt/dt = 1
    # Calculate values of dh/dt using odefunc
    dhdt = np.zeros((timesteps, num_points))
    for i in range(len(t)):
        for j in range(len(hidden)):
            # Ensure h_j has shape (1, 1) as this is expected by odefunc
            h_j = torch.Tensor([hidden[j]]).unsqueeze(0)
            dhdt[i, j] = odefunc(t[i], h_j)
    return t, hidden, dtdt, dhdt


def get_feature_history(trainer, dataloader, inputs, targets, num_epochs):
    """Helper function to record feature history while training a model. This is
    useful for visualizing the evolution of features.

    trainer : anode.training.Trainer instance

    dataloader : torch.utils.DataLoader

    inputs : torch.Tensor
        Tensor of shape (num_points, num_dims) containing a batch of data which
        will be used to visualize the evolution of the model.

    targets : torch.Tensor
        Shape (num_points, 1). The targets of the data in inputs.

    num_epochs : int
        Number of epochs to train for.
    """
    feature_history = []
    # Get features at beginning of training
    features, _ = trainer.model(inputs, return_features=True)
    feature_history.append(features.detach())

    for i in range(num_epochs):
        trainer.train(dataloader, 1)
        features, _ = trainer.model(inputs, return_features=True)
        feature_history.append(features.detach())

    return feature_history


def get_square_aspect_ratio(plt_axis):
    return np.diff(plt_axis.get_xlim())[0] / np.diff(plt_axis.get_ylim())[0]
