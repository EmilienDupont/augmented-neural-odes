import imageio
import os
from mpl_toolkits.mplot3d import Axes3D
from viz.plots import get_square_aspect_ratio


def feature_evolution_gif(feature_history, targets, dpi=100, alpha=0.5,
                          filename='feature_evolution.gif'):
    """Creates a gif of evolution of feature space. Works for 2 and 3
    dimensions.

    Parameters
    ----------
    feature_history : list of torch.Tensor

    targets : torch.Tensor

    dpi : int
        Controls resolution of gif.

    alpha : float
        Controls opacity of points.

    filename : string
        Gif will be saved to this filename.
    """
    if not filename.endswith(".gif"):
        raise RuntimeError("Filename must end in with .gif, but filename is {}".format(filename))
    base_filename = filename[:-4]

    # Color features by their target color
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    # Extract number of dimensions of features
    num_dims = feature_history[0].shape[1]

    for i, features in enumerate(feature_history):
        if num_dims == 2:
            # Plot features
            plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                        alpha=alpha, linewidths=0)
            # Remove all axes and ticks
            plt.tick_params(axis='both', which='both', bottom=False, top=False,
                            labelbottom=False, right=False, left=False,
                            labelleft=False)
            # Set square aspect ratio
            ax = plt.gca()
            ax.set_aspect(get_square_aspect_ratio(ax))
        elif num_dims == 3:
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(), features[:, 1].numpy(),
                       c=color, alpha=alpha, linewidths=0)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        # Save current frame
        plt.savefig(base_filename + "{}.png".format(i),
                    format='png', dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close()

    # Create gif based on saved images
    imgs = []
    for i in range(len(feature_history)):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        os.remove(img_file)  # remove img_file as we no longer need it
    imageio.mimwrite(filename, imgs)


def trajectory_gif(model, inputs, targets, timesteps, dpi=100, alpha=0.5,
                   alpha_line=0.3, filename='trajectory.gif'):
    """Creates a gif of input point trajectories according to model. Works for 2
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

    dpi : int
        Controls resolution of gif.

    alpha : float
        Controls opacity of points.

    alpha_line : float
        Controls opacity of lines.

    filename : string
        Gif will be saved to this filename.
    """
    if not filename.endswith(".gif"):
        raise RuntimeError("Filename must end in with .gif, but filename is {}".format(filename))
    base_filename = filename[:-4]

    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]

    # Calculate trajectories (timesteps, batch_size, input_dim)
    trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    # Extract number of dimensions of features
    num_dims = trajectories.shape[2]

    # Determine limits of plot from trajectories
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    if num_dims == 3:
        z_min, z_max = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
    # Add margins
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    if num_dims == 3:
        z_range = z_max - z_min
        z_min -= margin * z_range
        z_max += margin * z_range

    for t in range(timesteps):
        if num_dims == 2:
            plt.scatter(trajectories[t, :, 0].numpy(), trajectories[t, :, 1].numpy(), c=color,
                        alpha=alpha, linewidths=0)

            # For each point in batch, plot its trajectory
            if t > 0:
                for i in range(inputs.shape[0]):
                    trajectory = trajectories[:t + 1, i, :]
                    x_traj = trajectory[:, 0].numpy()
                    y_traj = trajectory[:, 1].numpy()
                    plt.plot(x_traj, y_traj, c=color[i], alpha=alpha_line)

            plt.tick_params(axis='both', which='both', bottom=False, top=False,
                            labelbottom=False, right=False, left=False, labelleft=False)

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            ax = plt.gca()
            ax.set_aspect(get_square_aspect_ratio(ax))
        elif num_dims == 3:
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.scatter(trajectories[t, :, 0].numpy(),
                       trajectories[t, :, 1].numpy(),
                       trajectories[t, :, 2].numpy(),
                       c=color, alpha=alpha, linewidths=0)
            # For each point in batch, plot its trajectory
            if t > 0:
                for i in range(inputs.shape[0]):
                    trajectory = trajectories[:t + 1, i, :]
                    x_traj = trajectory[:, 0].numpy()
                    y_traj = trajectory[:, 1].numpy()
                    z_traj = trajectory[:, 2].numpy()
                    ax.plot(x_traj, y_traj, z_traj, c=color[i], alpha=alpha)

            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.set_aspect(get_square_aspect_ratio(ax))

        # Save current frame
        plt.savefig(base_filename + "{}.png".format(t),
                    format='png', dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close()

    # Create gif based on saved images
    imgs = []
    for i in range(timesteps):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        os.remove(img_file)  # remove img_file as we no longer need it
    imageio.mimwrite(filename, imgs)
