import numpy as np
import torch


def svd_flex(tensor, svd_string, max_D=None, cutoff=1e-10, sv_right=True, sv_vec=None):
    """
    Split an input tensor into two pieces using a SVD across some partition

    Args:
        tensor (Tensor):    Pytorch tensor with at least two indices

        svd_string (str):   String of the form 'init_str->left_str,right_str',
                            where init_str describes the indices of tensor, and
                            left_str/right_str describe those of the left and
                            right output tensors. The characters of left_str
                            and right_str form a partition of the characters in
                            init_str, but each contain one additional character
                            representing the new bond which comes from the SVD

                            Reversing the terms in svd_string to the left and
                            right of '->' gives an ein_string which can be used
                            to multiply both output tensors to give a (low rank
                            approximation) of the input tensor

        cutoff (float):     A truncation threshold which eliminates any
                            singular values which are strictly less than cutoff

        max_D (int):        A maximum allowed value for the new bond. If max_D
                            is specified, the returned tensors

        sv_right (bool):    The SVD gives two orthogonal matrices and a matrix
                            of singular values. sv_right=True merges the SV
                            matrix with the right output, while sv_right=False
                            merges it with the left output

        sv_vec (Tensor):    Pytorch vector with length max_D, which is modified
                            in place to return the vector of singular values

    Returns:
        left_tensor (Tensor),
        right_tensor (Tensor):  Tensors whose indices are described by the
                                left_str and right_str parts of svd_string

        bond_dim:               The dimension of the new bond appearing from
                                the cutoff in our SVD. Note that this generally
                                won't match the dimension of left_/right_tensor
                                at this mode, which is padded with zeros
                                whenever max_D is specified
    """

    def prod(int_list):
        output = 1
        for num in int_list:
            output *= num
        return output

    with torch.no_grad():
        # Parse svd_string into init_str, left_str, and right_str
        svd_string = svd_string.replace(" ", "")
        init_str, post_str = svd_string.split("->")
        left_str, right_str = post_str.split(",")

        # Check formatting of init_str, left_str, and right_str
        assert all([c.islower() for c in init_str + left_str + right_str])
        assert len(set(init_str + left_str + right_str)) == len(init_str) + 1
        assert len(set(init_str)) + len(set(left_str)) + len(set(right_str)) == len(
            init_str
        ) + len(left_str) + len(right_str)

        # Get the special character representing our SVD-truncated bond
        bond_char = set(left_str).intersection(set(right_str)).pop()
        left_part = left_str.replace(bond_char, "")
        right_part = right_str.replace(bond_char, "")

        # Permute our tensor into something that can be viewed as a matrix
        ein_str = f"{init_str}->{left_part+right_part}"
        tensor = torch.einsum(ein_str, [tensor]).contiguous()

        left_shape = list(tensor.shape[: len(left_part)])
        right_shape = list(tensor.shape[len(left_part) :])
        left_dim, right_dim = prod(left_shape), prod(right_shape)

        tensor = tensor.view([left_dim, right_dim])

        # Get SVD and format so that left_mat * diag(svs) * right_mat = tensor
        left_mat, svs, right_mat = torch.svd(tensor)
        svs, _ = torch.sort(svs, descending=True)
        right_mat = torch.t(right_mat)

        # Decrease or increase our tensor sizes in the presence of max_D
        if max_D and len(svs) > max_D:
            svs = svs[:max_D]
            left_mat = left_mat[:, :max_D]
            right_mat = right_mat[:max_D]
        elif max_D and len(svs) < max_D:
            copy_svs = torch.zeros([max_D])
            copy_svs[: len(svs)] = svs
            copy_left = torch.zeros([left_mat.size(0), max_D])
            copy_left[:, : left_mat.size(1)] = left_mat
            copy_right = torch.zeros([max_D, right_mat.size(1)])
            copy_right[: right_mat.size(0)] = right_mat
            svs, left_mat, right_mat = copy_svs, copy_left, copy_right

        # If given as input, copy singular values into sv_vec
        if sv_vec is not None and svs.shape == sv_vec.shape:
            sv_vec[:] = svs
        elif sv_vec is not None and svs.shape != sv_vec.shape:
            raise TypeError(
                f"sv_vec.shape must be {list(svs.shape)}, but is "
                f"currently {list(sv_vec.shape)}"
            )

        # Find the truncation point relative to our singular value cutoff
        truncation = 0
        for s in svs:
            if s < cutoff:
                break
            truncation += 1
        if truncation == 0:
            raise RuntimeError(
                "SVD cutoff too large, attempted to truncate "
                "tensor to bond dimension 0"
            )

        # Perform the actual truncation
        if max_D:
            svs[truncation:] = 0
            left_mat[:, truncation:] = 0
            right_mat[truncation:] = 0
        else:
            # If max_D wasn't given, set it to the truncation index
            max_D = truncation
            svs = svs[:truncation]
            left_mat = left_mat[:, :truncation]
            right_mat = right_mat[:truncation]

        # Merge the singular values into the appropriate matrix
        if sv_right:
            right_mat = torch.einsum("l,lr->lr", [svs, right_mat])
        else:
            left_mat = torch.einsum("lr,r->lr", [left_mat, svs])

        # Reshape the matrices to make them proper tensors
        left_tensor = left_mat.view(left_shape + [max_D])
        right_tensor = right_mat.view([max_D] + right_shape)

        # Finally, permute the indices into the desired order
        if left_str != left_part + bond_char:
            left_tensor = torch.einsum(
                f"{left_part+bond_char}->{left_str}", [left_tensor]
            )
        if right_str != bond_char + right_part:
            right_tensor = torch.einsum(
                f"{bond_char+right_part}->{right_str}", [right_tensor]
            )

        return left_tensor, right_tensor, truncation


def init_tensor(shape, bond_str, init_method):
    """
    Initialize a tensor with a given shape

    Args:
        shape:       The shape of our output parameter tensor.

        bond_str:    The bond string describing our output parameter tensor,
                     which is used in 'random_eye' initialization method.
                     The characters 'l' and 'r' are used to refer to the
                     left or right virtual indices of our tensor, and are
                     both required to be present for the random_eye and
                     min_random_eye initialization methods.

        init_method: The method used to initialize the entries of our tensor.
                     This can be either a string, or else a tuple whose first
                     entry is an initialization method and whose remaining
                     entries are specific to that method. In each case, std
                     will always refer to a standard deviation for a random
                     normal random component of each entry of the tensor.

                     Allowed options are:
                        * ('random_eye', std): Initialize each tensor input
                            slice close to the identity
                        * ('random_zero', std): Initialize each tensor input
                            slice close to the zero matrix
                        * ('min_random_eye', std, init_dim): Initialize each
                            tensor input slice close to a truncated identity
                            matrix, whose truncation leaves init_dim unit
                            entries on the diagonal. If init_dim is larger
                            than either of the bond dimensions, then init_dim
                            is capped at the smaller bond dimension.
    """
    # Unpack init_method if it is a tuple
    if not isinstance(init_method, str):
        init_str = init_method[0]
        std = init_method[1]
        if init_str == "min_random_eye":
            init_dim = init_method[2]

        init_method = init_str
    else:
        std = 1e-9

    # Check that bond_str is properly sized and doesn't have repeat indices
    assert len(shape) == len(bond_str)
    assert len(set(bond_str)) == len(bond_str)

    if init_method not in ["random_eye", "min_random_eye", "random_zero"]:
        raise ValueError(f"Unknown initialization method: {init_method}")

    if init_method in ["random_eye", "min_random_eye"]:
        bond_chars = ["l", "r"]
        assert all([c in bond_str for c in bond_chars])

        # Initialize our tensor slices as identity matrices which each fill
        # some or all of the initially allocated bond space
        if init_method == "min_random_eye":

            # The dimensions for our initial identity matrix. These will each
            # be init_dim, unless init_dim exceeds one of the bond dimensions
            bond_dims = [shape[bond_str.index(c)] for c in bond_chars]
            if all([init_dim <= full_dim for full_dim in bond_dims]):
                bond_dims = [init_dim, init_dim]
            else:
                init_dim = min(bond_dims)

            eye_shape = [init_dim if c in bond_chars else 1 for c in bond_str]
            expand_shape = [
                init_dim if c in bond_chars else shape[i]
                for i, c in enumerate(bond_str)
            ]

        elif init_method == "random_eye":
            eye_shape = [
                shape[i] if c in bond_chars else 1 for i, c in enumerate(bond_str)
            ]
            expand_shape = shape
            bond_dims = [shape[bond_str.index(c)] for c in bond_chars]

        eye_tensor = torch.eye(bond_dims[0], bond_dims[1]).view(eye_shape)
        eye_tensor = eye_tensor.expand(expand_shape)

        tensor = torch.zeros(shape)
        tensor[[slice(dim) for dim in expand_shape]] = eye_tensor

        # Add on a bit of random noise
        tensor += std * torch.randn(shape)

    elif init_method == "random_zero":
        tensor = std * torch.randn(shape)

    return tensor


### OLDER MISCELLANEOUS FUNCTIONS ###   # noqa: E266


def onehot(labels, max_value):
    """
    Convert a batch of labels from the set {0, 1,..., num_value-1} into their
    onehot encoded counterparts
    """
    label_vecs = torch.zeros([len(labels), max_value])

    for i, label in enumerate(labels):
        label_vecs[i, label] = 1.0

    return label_vecs


def joint_shuffle(input_data, input_labels):
    """
    Shuffle input data and labels in a joint manner, so each label points to
    its corresponding datum. Works for both regular and CUDA tensors
    """
    assert input_data.is_cuda == input_labels.is_cuda
    use_gpu = input_data.is_cuda
    if use_gpu:
        input_data, input_labels = input_data.cpu(), input_labels.cpu()

    data, labels = input_data.numpy(), input_labels.numpy()

    # Shuffle relative to the same seed
    np.random.seed(0)
    np.random.shuffle(data)
    np.random.seed(0)
    np.random.shuffle(labels)

    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    if use_gpu:
        data, labels = data.cuda(), labels.cuda()

    return data, labels


def load_HV_data(length):
    """
    Output a toy "horizontal/vertical" data set of black and white
    images with size length x length. Each image contains a single
    horizontal or vertical stripe, set against a background
    of the opposite color. The labels associated with these images
    are either 0 (horizontal stripe) or 1 (vertical stripe).

    In its current version, this returns two data sets, a training
    set with 75% of the images and a test set with 25% of the
    images.
    """
    num_images = 4 * (2 ** (length - 1) - 1)
    num_patterns = num_images // 2
    split = num_images // 4

    if length > 14:
        print(
            "load_HV_data will generate {} images, "
            "this could take a while...".format(num_images)
        )

    images = np.empty([num_images, length, length], dtype=np.float32)
    labels = np.empty(num_images, dtype=np.int)

    # Used to generate the stripe pattern from integer i below
    template = "{:0" + str(length) + "b}"

    for i in range(1, num_patterns + 1):
        pattern = template.format(i)
        pattern = [int(s) for s in pattern]

        for j, val in enumerate(pattern):
            # Horizontal stripe pattern
            images[2 * i - 2, j, :] = val
            # Vertical stripe pattern
            images[2 * i - 1, :, j] = val

        labels[2 * i - 2] = 0
        labels[2 * i - 1] = 1

    # Shuffle and partition into training and test sets
    np.random.seed(0)
    np.random.shuffle(images)
    np.random.seed(0)
    np.random.shuffle(labels)

    train_images, train_labels = images[split:], labels[split:]
    test_images, test_labels = images[:split], labels[:split]

    return (
        torch.from_numpy(train_images),
        torch.from_numpy(train_labels),
        torch.from_numpy(test_images),
        torch.from_numpy(test_labels),
    )
