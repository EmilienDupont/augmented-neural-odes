"""
TODO:
    (1) Update master to include all the new features in dynamic_capacity
"""
import torch
import torch.nn as nn

from .utils import init_tensor, svd_flex
from .contractables import (
    SingleMat,
    MatRegion,
    OutputCore,
    ContractableList,
    EdgeVec,
    OutputMat,
)


class TI_MPS(nn.Module):
    """
    Sequence MPS which converts input of arbitrary length to a single output vector
    """

    def __init__(
        self,
        output_dim,
        bond_dim,
        feature_dim=2,
        parallel_eval=False,
        fixed_ends=False,
        init_std=1e-9,
        use_bias=True,
        fixed_bias=True,
    ):
        super().__init__()

        # Initialize the core tensor defining our model near the identity
        # This tensor holds all of the trainable parameters of our model
        tensor = init_tensor(
            bond_str="lri",
            shape=[bond_dim, bond_dim, feature_dim],
            init_method=("random_zero", init_std),
        )
        self.register_parameter(name="core_tensor", param=nn.Parameter(tensor))

        # Define our initial vector and terminal matrix, which are both
        # functional modules, i.e. unchanged during training
        assert isinstance(fixed_ends, bool)
        self.init_vector = InitialVector(bond_dim, fixed_vec=fixed_ends)
        self.terminal_mat = TerminalOutput(bond_dim, output_dim, fixed_mat=fixed_ends)

        # Set the bias matrix
        if use_bias:
            # bias_mat is identity when fixed_bias=True, near-identity otherwise
            if fixed_bias:
                bias_mat = torch.eye(bond_dim)
                self.register_buffer(name="bias_mat", tensor=bias_mat)
            else:
                bias_mat = init_tensor(
                    bond_str="lr",
                    shape=[bond_dim, bond_dim],
                    init_method=("random_eye", init_std),
                )
                self.register_parameter(name="bias_mat", param=nn.Parameter(bias_mat))
        else:
            self.bias_mat = None

        # Set the rest of our TI_MPS attributes
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.parallel_eval = parallel_eval
        self.use_bias = use_bias
        self.fixed_bias = fixed_bias
        self.feature_map = None

    def forward(self, input_data):
        """
        Converts batch input tensor into a batch output tensor

        Args:
            input_data: A tensor of shape [batch_size, length, feature_dim].
        """

        # Reformat our input to a batch format, padding with zeros as needed
        batch_input = self.format_input(input_data)
        batch_size = batch_input.size(0)
        seq_len = batch_input.size(1)

        # Build up a contractable_list as EdgeVec + MatRegion + OutputMat
        expanded_core = self.core_tensor.expand(
            [seq_len, self.bond_dim, self.bond_dim, self.feature_dim]
        )
        input_region = InputRegion(
            expanded_core,
            use_bias=self.use_bias,
            fixed_bias=self.fixed_bias,
            bias_mat=self.bias_mat,
            ephemeral=True,
        )
        contractable_list = [input_region(batch_input)]

        # Prepend an EdgeVec and append an OutputMat
        contractable_list = [self.init_vector()] + contractable_list
        contractable_list.append(self.terminal_mat())

        # Wrap contractable_list as a ContractableList instance
        contractable_list = ContractableList(contractable_list)

        # Contract everything in contractable_list
        output = contractable_list.reduce(parallel_eval=self.parallel_eval)
        batch_output = output.tensor

        # Check shape before returning output values
        assert output.bond_str == "bo"
        assert batch_output.size(0) == batch_size
        assert batch_output.size(1) == self.output_dim

        return batch_output

    def format_input(self, input_data):
        """
        Converts input list of sequences into a single batch sequence tensor.

        If input is already a batch tensor, it is returned unchanged. Otherwise,
        convert input list into a batch sequence with shape [batch_size, length,
        feature_dim].

        If self.use_bias = self.fixed_bias = True, then sequences of different
        lengths can be used, in which case shorter sequences are padded with
        zeros at the end, making the batch tensor length equal to the length
        of the longest input sequence.

        Args:
            input_data: A tensor of shape [batch_size, length] or
            [batch_size, length, feature_dim], or a list of length batch_size,
            whose i'th item is a tensor of shape [length_i, feature_dim] or
            [length_i]. If self.use_bias or self.fixed_bias are False, then
            length_i must be the same for all i.
        """
        feature_dim = self.feature_dim

        # If we get a batch tensor, just embed it and/or return it unchanged
        if isinstance(input_data, torch.Tensor):
            if len(input_data.shape) == 2:
                input_data = self.embed_input(input_data)

            # Check to make sure shape is alright
            shape = input_data.shape
            assert len(shape) == 3
            assert shape[2] == feature_dim

            return input_data

        # Collate the input list into a single batch tensor
        elif isinstance(input_data, list):
            # Check formatting, require that input sequences are either all
            # unembedded or all pre-embedded
            num_modes = len(input_data[0].shape)
            assert num_modes in [1, 2]
            assert all(
                [
                    isinstance(s, torch.Tensor) and len(s.shape) == num_modes
                    for s in input_data
                ]
            )
            assert num_modes == 1 or all([s.size(1) == feature_dim for s in input_data])

            # Check that all the sequences are the same length or can be padded
            max_len = max([s.size(0) for s in input_data])
            can_pad = self.use_bias and self.fixed_bias
            if not can_pad and any([s.size(0) != max_len for s in input_data]):
                raise ValueError(
                    "To process input_data as list of sequences "
                    "with different lengths, must have self.use_bias="
                    "self.fixed_bias=True (currently self.use_bias="
                    f"{self.use_bias}, self.fixed_bias={self.fixed_bias})"
                )

            # Pad the sequences with zeros (if needed), return as batch tensor
            if can_pad:
                batch_size = len(input_data)
                full_size = [batch_size, max_len, feature_dim]
                batch_input = torch.zeros(full_size[: num_modes + 1])

                # Copy each sequence into batch_input
                for i, seq in enumerate(input_data):
                    batch_input[i, : seq.size(0)] = seq
            else:
                batch_input = torch.stack(input_data)

            # Embed everything (if needed) and return the batch tensor
            if len(batch_input.shape) == 2:
                batch_input = self.embed_input(batch_input)

            return batch_input

        else:
            raise ValueError(
                "input_data must either be Tensor with shape"
                "[batch_size, length] or [batch_size, length, feature_dim], "
                "or list of Tensors with shapes [length_i, feature_dim] or "
                "[length_i]"
            )

    def embed_input(self, input_data):
        """
        Embed pixels of input_data into separate local feature spaces

        Args:
            input_data (Tensor):    Input with shape [batch_size, length].

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                    [batch_size, input_dim, feature_dim]
        """
        assert len(input_data.shape) == 2

        # Get relevant dimensions
        batch_dim, length = input_data.shape
        feature_dim = self.feature_dim
        embedded_shape = [batch_dim, length, feature_dim]

        # Apply a custom embedding map if it has been defined by the user
        if self.feature_map is not None:
            f_map = self.feature_map
            embedded_data = torch.stack(
                [torch.stack([f_map(x) for x in batch]) for batch in input_data]
            )

            # Make sure our embedded input has the desired size
            assert list(embedded_data.shape) == embedded_shape

        # Otherwise, use a simple linear embedding map with feature_dim = 2
        else:
            if self.feature_dim != 2:
                raise RuntimeError(
                    f"self.feature_dim = {feature_dim}, but "
                    "default feature_map requires self.feature_dim = 2"
                )
            embedded_data = torch.empty(embedded_shape)

            embedded_data[:, :, 0] = input_data
            embedded_data[:, :, 1] = 1 - input_data

        return embedded_data

    def register_feature_map(self, feature_map):
        """
        Register a custom feature map to be used for embedding input data

        Args:
            feature_map (function): Takes a single scalar input datum and
                                    returns an embedded representation of the
                                    image. The output size of the function must
                                    match self.feature_dim. If feature_map=None,
                                    then the feature map will be reset to a
                                    simple default linear embedding
        """
        if feature_map is not None:
            # Test to make sure feature_map outputs vector of proper size
            test_out = feature_map(torch.tensor(0))
            assert isinstance(test_out, torch.Tensor)

            out_shape, needed_shape = list(test_out.shape), [self.feature_dim]
            if out_shape != needed_shape:
                raise ValueError(
                    "Given feature_map returns values with shape "
                    f"{list(out_shape)}, but should return "
                    f"values of size {list(needed_shape)}"
                )

        self.feature_map = feature_map


class MPS(nn.Module):
    """
    Tunable MPS model giving mapping from fixed-size data to output vector

    Model works by first converting each 'pixel' (local data) to feature
    vector via a simple embedding, then contracting embeddings with inputs
    to each MPS cores. The resulting transition matrices are contracted
    together along bond dimensions (i.e. hidden state spaces), with output
    produced via an uncontracted edge of an additional output core.

    MPS model permits many customizable behaviors, including custom
    'routing' of MPS through the input, choice of boundary conditions
    (meaning the model can act as a tensor train or a tensor ring),
    GPU-friendly parallel evaluation, and an experimental mode to support
    adaptive choice of bond dimensions based on singular value spectrum.

    Args:
        input_dim:       Number of 'pixels' in the input to the MPS
        output_dim:      Size of the vectors output by MPS via output core
        bond_dim:        Dimension of the 'bonds' connecting adjacent MPS
                         cores, which act as hidden state spaces of the
                         model. In adaptive mode, bond_dim instead
                         specifies the maximum allowed bond dimension
        feature_dim:     Size of the local feature spaces each pixel is
                         embedded into (default: 2)
        periodic_bc:     Whether MPS has periodic boundary conditions (i.e.
                         is a tensor ring) or open boundary conditions
                         (i.e. is a tensor train) (default: False)
        parallel_eval:   Whether contraction of tensors is performed in a
                         serial or parallel fashion. The former is less
                         expensive for open boundary conditions, but
                         parallelizes more poorly (default: False)
        label_site:      Location in the MPS chain where output is placed
                         (default: input_dim // 2)
        path:            List specifying a path through the input data
                         which MPS is 'routed' along. For example, choosing
                         path=[0, 1, ..., input_dim-1] gives a standard
                         in-order traversal (behavior when path=None), while
                         path=[0, 2, ..., input_dim-1] specifies an MPS
                         accepting input only from even-valued input pixels
                         (default: None)
        init_std:        Size of the Gaussian noise used in default
                         near-identity initialization (default: 1e-9)
        initializer:     Pytorch initializer for custom initialization of
                         MPS cores, with None specifying default
                         near-identity initialization (default: None)
        use_bias:        Whether to use trainable bias matrices in MPS
                         cores, which are initialized near the zero matrix
                         (default: False)
        adaptive_mode:   Whether MPS is trained with experimental adaptive
                         bond dimensions selection (default: False)
        cutoff:          Singular value cutoff controlling bond dimension
                         adaptive selection (default: 1e-9)
        merge_threshold: Number of inputs before adaptive MPS shifts its
                         merge state once, with two shifts leading to the
                         update of all bond dimensions (default: 2000)
    """

    # TODO: Support arbitrary initializers
    # TODO: Clean up the current treatment of initialization
    # TODO: Resolve weirdness with fixed bias and initialization choice
    # TODO: Add function to convert to canonical form
    # TODO: Fix issue of no training when use_bias=False

    def __init__(
        self,
        input_dim,
        output_dim,
        bond_dim,
        feature_dim=2,
        periodic_bc=False,
        parallel_eval=False,
        label_site=None,
        path=None,
        init_std=1e-9,
        initializer=None,
        use_bias=True,
        adaptive_mode=False,
        cutoff=1e-10,
        merge_threshold=2000,
    ):
        super().__init__()

        if label_site is None:
            label_site = input_dim // 2
        assert label_site >= 0 and label_site <= input_dim

        # Using bias matrices in adaptive_mode is too complicated, so I'm
        # disabling it here
        if adaptive_mode:
            use_bias = False

        # Our MPS is made of two InputRegions separated by an OutputSite.
        module_list = []
        init_args = {
            "bond_str": "slri",
            "shape": [label_site, bond_dim, bond_dim, feature_dim],
            "init_method": (
                "min_random_eye" if adaptive_mode else "random_zero",
                init_std,
                output_dim,
            ),
        }

        # The first input region
        if label_site > 0:
            tensor = init_tensor(**init_args)

            module_list.append(InputRegion(tensor, use_bias=use_bias, fixed_bias=False))

        # The output site
        tensor = init_tensor(
            shape=[output_dim, bond_dim, bond_dim],
            bond_str="olr",
            init_method=(
                "min_random_eye" if adaptive_mode else "random_eye",
                init_std,
                output_dim,
            ),
        )
        module_list.append(OutputSite(tensor))

        # The other input region
        if label_site < input_dim:
            init_args["shape"] = [
                input_dim - label_site,
                bond_dim,
                bond_dim,
                feature_dim,
            ]
            tensor = init_tensor(**init_args)
            module_list.append(InputRegion(tensor, use_bias=use_bias, fixed_bias=False))

        # Initialize linear_region according to our adaptive_mode specification
        if adaptive_mode:
            self.linear_region = MergedLinearRegion(
                module_list=module_list,
                periodic_bc=periodic_bc,
                parallel_eval=parallel_eval,
                cutoff=cutoff,
                merge_threshold=merge_threshold,
            )

            # Initialize the list of bond dimensions, which starts out constant
            self.bond_list = bond_dim * torch.ones(input_dim + 2, dtype=torch.long)
            if not periodic_bc:
                self.bond_list[0], self.bond_list[-1] = 1, 1

            # Initialize the list of singular values, which start out at -1
            self.sv_list = -1.0 * torch.ones([input_dim + 2, bond_dim])

        else:
            self.linear_region = LinearRegion(
                module_list=module_list,
                periodic_bc=periodic_bc,
                parallel_eval=parallel_eval,
            )
        assert len(self.linear_region) == input_dim

        if path:
            assert isinstance(path, (list, torch.Tensor))
            assert len(path) == input_dim

        # Set the rest of our MPS attributes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.feature_dim = feature_dim
        self.periodic_bc = periodic_bc
        self.adaptive_mode = adaptive_mode
        self.label_site = label_site
        self.path = path
        self.use_bias = use_bias
        self.cutoff = cutoff
        self.merge_threshold = merge_threshold
        self.feature_map = None

    def forward(self, input_data):
        """
        Embed our data and pass it to an MPS with a single output site

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim] or
                                 [batch_size, input_dim, feature_dim]. In the
                                 former case, the data points are turned into
                                 2D vectors using a default linear feature map.

                                 When using a user-specified path, the size of
                                 the second tensor mode need not exactly equal
                                 input_dim, since the path variable is used to
                                 slice a certain subregion of input_data. This
                                 can be used to define multiple MPS 'strings',
                                 which act on different parts of the input.
        """
        # For custom paths, rearrange our input into the desired order
        if self.path:
            path_inputs = []
            for site_num in self.path:
                path_inputs.append(input_data[:, site_num])
            input_data = torch.stack(path_inputs, dim=1)

        # Embed our input data before feeding it into our linear region
        input_data = self.embed_input(input_data)
        output = self.linear_region(input_data)

        # If we got a tuple as output, then use the last two entries to
        # update our bond dimensions and singular values
        if isinstance(output, tuple):
            output, new_bonds, new_svs = output

            assert len(new_bonds) == len(self.bond_list)
            assert len(new_bonds) == len(new_svs)
            for i, bond_dim in enumerate(new_bonds):
                if bond_dim is not None:
                    assert new_svs[i] is not None
                    self.bond_list[i] = bond_dim
                    self.sv_list[i] = new_svs[i]

        return output

    def embed_input(self, input_data):
        """
        Embed pixels of input_data into separate local feature spaces

        Args:
            input_data (Tensor):    Input with shape [batch_size, input_dim], or
                                    [batch_size, input_dim, feature_dim]. In the
                                    latter case, the data is assumed to already
                                    be embedded, and is returned unchanged.

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                    [batch_size, input_dim, feature_dim]
        """
        assert len(input_data.shape) in [2, 3]
        assert input_data.size(1) == self.input_dim

        # If input already has a feature dimension, return it as is
        if len(input_data.shape) == 3:
            if input_data.size(2) != self.feature_dim:
                raise ValueError(
                    f"input_data has wrong shape to be unembedded "
                    "or pre-embedded data (input_data.shape = "
                    f"{list(input_data.shape)}, feature_dim = {self.feature_dim})"
                )
            return input_data

        # Apply a custom embedding map if it has been defined by the user
        if self.feature_map is not None:
            f_map = self.feature_map
            embedded_data = torch.stack(
                [torch.stack([f_map(x) for x in batch]) for batch in input_data]
            )

            # Make sure our embedded input has the desired size
            assert embedded_data.shape == torch.Size(
                [input_data.size(0), self.input_dim, self.feature_dim]
            )

        # Otherwise, use a simple linear embedding map with feature_dim = 2
        else:
            if self.feature_dim != 2:
                raise RuntimeError(
                    f"self.feature_dim = {self.feature_dim}, "
                    "but default feature_map requires self.feature_dim = 2"
                )

            embedded_data = torch.stack([input_data, 1 - input_data], dim=2)

        return embedded_data

    def register_feature_map(self, feature_map):
        """
        Register a custom feature map to be used for embedding input data

        Args:
            feature_map (function): Takes a single scalar input datum and
                                    returns an embedded representation of the
                                    image. The output size of the function must
                                    match self.feature_dim. If feature_map=None,
                                    then the feature map will be reset to a
                                    simple default linear embedding
        """
        if feature_map is not None:
            # Test to make sure feature_map outputs vector of proper size
            out_shape = feature_map(torch.tensor(0)).shape
            needed_shape = torch.Size([self.feature_dim])
            if out_shape != needed_shape:
                raise ValueError(
                    "Given feature_map returns values of size "
                    f"{list(out_shape)}, but should return "
                    f"values of size {list(needed_shape)}"
                )

        self.feature_map = feature_map

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return self.linear_region.core_len()

    def __len__(self):
        """
        Returns the number of input sites, which equals the input size
        """
        return self.input_dim


class LinearRegion(nn.Module):
    """
    List of modules which feeds input to each module and returns reduced output
    """

    def __init__(
        self, module_list, periodic_bc=False, parallel_eval=False, module_states=None
    ):
        # Check that module_list is a list whose entries are Pytorch modules
        if not isinstance(module_list, list) or module_list is []:
            raise ValueError("Input to LinearRegion must be nonempty list")
        for i, item in enumerate(module_list):
            if not isinstance(item, nn.Module):
                raise ValueError(
                    "Input items to LinearRegion must be PyTorch "
                    f"Module instances, but item {i} is not"
                )
        super().__init__()

        # Wrap as a ModuleList for proper parameter registration
        self.module_list = nn.ModuleList(module_list)
        self.periodic_bc = periodic_bc
        self.parallel_eval = parallel_eval

    def forward(self, input_data):
        """
        Contract input with list of MPS cores and return result as contractable

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                                   feature_dim]
        """
        # Check that input_data has the correct shape
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        periodic_bc = self.periodic_bc
        parallel_eval = self.parallel_eval
        lin_bonds = ["l", "r"]

        # Whether to move intermediate vectors to a GPU (fixes Issue #8)
        to_cuda = input_data.is_cuda
        device = f"cuda:{input_data.get_device()}" if to_cuda else "cpu"

        # For each module, pull out the number of pixels needed and call that
        # module's forward() method, putting the result in contractable_list
        ind = 0
        contractable_list = []
        for module in self.module_list:
            mod_len = len(module)
            if mod_len == 1:
                mod_input = input_data[:, ind]
            else:
                mod_input = input_data[:, ind : (ind + mod_len)]
            ind += mod_len

            contractable_list.append(module(mod_input))

        # For periodic boundary conditions, reduce contractable_list and
        # trace over the left and right indices to get our output
        if periodic_bc:
            contractable_list = ContractableList(contractable_list)
            contractable = contractable_list.reduce(parallel_eval=True)

            # Unpack the output (atomic) contractable
            tensor, bond_str = contractable.tensor, contractable.bond_str
            assert all(c in bond_str for c in lin_bonds)

            # Build einsum string for the trace of tensor
            in_str, out_str = "", ""
            for c in bond_str:
                if c in lin_bonds:
                    in_str += "l"
                else:
                    in_str += c
                    out_str += c
            ein_str = in_str + "->" + out_str

            # Return the trace over left and right indices
            return torch.einsum(ein_str, [tensor])

        # For open boundary conditions, add dummy edge vectors to
        # contractable_list and reduce everything to get our output
        else:
            # Get the dimension of left and right bond indices
            end_items = [contractable_list[i] for i in [0, -1]]
            bond_strs = [item.bond_str for item in end_items]
            bond_inds = [bs.index(c) for (bs, c) in zip(bond_strs, lin_bonds)]
            bond_dims = [
                item.tensor.size(ind) for (item, ind) in zip(end_items, bond_inds)
            ]

            # Build dummy end vectors and insert them at the ends of our list
            end_vecs = [torch.zeros(dim).to(device) for dim in bond_dims]

            for vec in end_vecs:
                vec[0] = 1
            contractable_list.insert(0, EdgeVec(end_vecs[0], is_left_vec=True))
            contractable_list.append(EdgeVec(end_vecs[1], is_left_vec=False))

            # Multiply together everything in contractable_list
            contractable_list = ContractableList(contractable_list)
            output = contractable_list.reduce(parallel_eval=parallel_eval)

            return output.tensor

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.core_len() for module in self.module_list])

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return sum([len(module) for module in self.module_list])


class MergedLinearRegion(LinearRegion):
    """
    Dynamic variant of LinearRegion that periodically rearranges its submodules
    """

    def __init__(
        self,
        module_list,
        periodic_bc=False,
        parallel_eval=False,
        cutoff=1e-10,
        merge_threshold=2000,
    ):
        # Initialize a LinearRegion with our given module_list
        super().__init__(module_list, periodic_bc, parallel_eval)

        # Initialize attributes self.module_list_0 and self.module_list_1
        # using the unmerged self.module_list, then redefine the latter in
        # terms of one of the former lists
        self.offset = 0
        self._merge(offset=self.offset)
        self._merge(offset=(self.offset + 1) % 2)
        self.module_list = getattr(self, f"module_list_{self.offset}")

        # Initialize variables used during switching
        self.input_counter = 0
        self.merge_threshold = merge_threshold
        self.cutoff = cutoff

    def forward(self, input_data):
        """
        Contract input with list of MPS cores and return result as contractable

        MergedLinearRegion keeps an input counter of the number of inputs, and
        when this exceeds its merge threshold, triggers an unmerging and
        remerging of its parameter tensors.

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                                   feature_dim]
        """
        # If we've hit our threshold, flip the merge state of our tensors
        if self.input_counter >= self.merge_threshold:
            bond_list, sv_list = self._unmerge(cutoff=self.cutoff)
            self.offset = (self.offset + 1) % 2
            self._merge(offset=self.offset)
            self.input_counter -= self.merge_threshold

            # Point self.module_list to the appropriate merged module
            self.module_list = getattr(self, f"module_list_{self.offset}")
        else:
            bond_list, sv_list = None, None

        # Increment our counter and call the LinearRegion's forward method
        self.input_counter += input_data.size(0)
        output = super().forward(input_data)

        # If we flipped our merge state, then return the bond_list and output
        if bond_list:
            return output, bond_list, sv_list
        else:
            return output

    @torch.no_grad()
    def _merge(self, offset):
        """
        Convert unmerged modules in self.module_list to merged counterparts

        Calling _merge (or _unmerge) directly can cause undefined behavior,
        but see MergedLinearRegion.forward for intended use

        This proceeds by first merging all unmerged cores internally, then
        merging lone cores when possible during a second sweep
        """
        assert offset in [0, 1]

        unmerged_list = self.module_list

        # Merge each core internally and add the results to midway_list
        site_num = offset
        merged_list = []
        for core in unmerged_list:
            assert not isinstance(core, MergedInput)
            assert not isinstance(core, MergedOutput)

            # Apply internal merging routine if our core supports it
            if hasattr(core, "_merge"):
                merged_list.extend(core._merge(offset=site_num % 2))
            else:
                merged_list.append(core)

            site_num += core.core_len()

        # Merge pairs of cores when possible (currently only with
        # InputSites), making sure to respect the offset for merging.
        while True:
            mod_num, site_num = 0, 0
            combined_list = []

            while mod_num < len(merged_list) - 1:
                left_core, right_core = merged_list[mod_num : mod_num + 2]
                new_core = self.combine(left_core, right_core, merging=True)

                # If cores aren't combinable, move our sliding window by 1
                if new_core is None or offset != site_num % 2:
                    combined_list.append(left_core)
                    mod_num += 1
                    site_num += left_core.core_len()

                # If we get something new, move to the next distinct pair
                else:
                    assert (
                        new_core.core_len()
                        == left_core.core_len() + right_core.core_len()
                    )
                    combined_list.append(new_core)
                    mod_num += 2
                    site_num += new_core.core_len()

                # Add the last core if there's nothing to merge it with
                if mod_num == len(merged_list) - 1:
                    combined_list.append(merged_list[mod_num])
                    mod_num += 1

            # We're finished when unmerged_list remains unchanged
            if len(combined_list) == len(merged_list):
                break
            else:
                merged_list = combined_list

        # Finally, update the appropriate merged module list
        list_name = f"module_list_{offset}"
        # If the merged module list hasn't been set yet, initialize it
        if not hasattr(self, list_name):
            setattr(self, list_name, nn.ModuleList(merged_list))

        # Otherwise, do an in-place update so that all tensors remain
        # properly registered with whatever optimizer we use
        else:
            module_list = getattr(self, list_name)
            assert len(module_list) == len(merged_list)
            for i in range(len(module_list)):
                assert module_list[i].tensor.shape == merged_list[i].tensor.shape
                module_list[i].tensor[:] = merged_list[i].tensor

    @torch.no_grad()
    def _unmerge(self, cutoff=1e-10):
        """
        Convert merged modules to unmerged counterparts

        Calling _unmerge (or _merge) directly can cause undefined behavior,
        but see MergedLinearRegion.forward for intended use

        This proceeds by first unmerging all merged cores internally, then
        combining lone cores where possible
        """
        list_name = f"module_list_{self.offset}"
        merged_list = getattr(self, list_name)

        # Unmerge each core internally and add results to unmerged_list
        unmerged_list, bond_list, sv_list = [], [None], [None]
        for core in merged_list:

            # Apply internal unmerging routine if our core supports it
            if hasattr(core, "_unmerge"):
                new_cores, new_bonds, new_svs = core._unmerge(cutoff)
                unmerged_list.extend(new_cores)
                bond_list.extend(new_bonds[1:])
                sv_list.extend(new_svs[1:])
            else:
                assert not isinstance(core, InputRegion)
                unmerged_list.append(core)
                bond_list.append(None)
                sv_list.append(None)

        # Combine all combinable pairs of cores. This occurs in several
        # passes, and for now acts nontrivially only on InputSite instances
        while True:
            mod_num = 0
            combined_list = []

            while mod_num < len(unmerged_list) - 1:
                left_core, right_core = unmerged_list[mod_num : mod_num + 2]
                new_core = self.combine(left_core, right_core, merging=False)

                # If cores aren't combinable, move our sliding window by 1
                if new_core is None:
                    combined_list.append(left_core)
                    mod_num += 1

                # If we get something new, move to the next distinct pair
                else:
                    combined_list.append(new_core)
                    mod_num += 2

                # Add the last core if there's nothing to combine it with
                if mod_num == len(unmerged_list) - 1:
                    combined_list.append(unmerged_list[mod_num])
                    mod_num += 1

            # We're finished when unmerged_list remains unchanged
            if len(combined_list) == len(unmerged_list):
                break
            else:
                unmerged_list = combined_list

        # Find the average (log) norm of all of our cores
        log_norms = []
        for core in unmerged_list:
            log_norms.append([torch.log(norm) for norm in core.get_norm()])
        log_scale = sum([sum(ns) for ns in log_norms])
        log_scale /= sum([len(ns) for ns in log_norms])

        # Now rescale all cores so that their norms are roughly equal
        scales = [[torch.exp(log_scale - n) for n in ns] for ns in log_norms]
        for core, these_scales in zip(unmerged_list, scales):
            core.rescale_norm(these_scales)

        # Add our unmerged module list as a new attribute and return
        # the updated bond dimensions
        self.module_list = nn.ModuleList(unmerged_list)
        return bond_list, sv_list

    def combine(self, left_core, right_core, merging):
        """
        Combine a pair of cores into a new core using context-dependent rules

        Depending on the types of left_core and right_core, along with whether
        we're currently merging (merging=True) or unmerging (merging=False),
        either return a new core, or None if no rule exists for this context
        """

        # Combine an OutputSite with a stray InputSite, return a MergedOutput
        if merging and (
            (isinstance(left_core, OutputSite) and isinstance(right_core, InputSite))
            or (isinstance(left_core, InputSite) and isinstance(right_core, OutputSite))
        ):

            left_site = isinstance(left_core, InputSite)
            if left_site:
                new_tensor = torch.einsum(
                    "lui,our->olri", [left_core.tensor, right_core.tensor]
                )
            else:
                new_tensor = torch.einsum(
                    "olu,uri->olri", [left_core.tensor, right_core.tensor]
                )
            return MergedOutput(new_tensor, left_output=(not left_site))

        # Combine an InputRegion with a stray InputSite, return an InputRegion
        elif not merging and (
            (isinstance(left_core, InputRegion) and isinstance(right_core, InputSite))
            or (
                isinstance(left_core, InputSite) and isinstance(right_core, InputRegion)
            )
        ):

            left_site = isinstance(left_core, InputSite)
            if left_site:
                left_tensor = left_core.tensor.unsqueeze(0)
                right_tensor = right_core.tensor
            else:
                left_tensor = left_core.tensor
                right_tensor = right_core.tensor.unsqueeze(0)

            assert left_tensor.shape[1:] == right_tensor.shape[1:]
            new_tensor = torch.cat([left_tensor, right_tensor])

            return InputRegion(new_tensor)

        # If this situation doesn't belong to the above cases, return None
        else:
            return None

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.core_len() for module in self.module_list])

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return sum([len(module) for module in self.module_list])


class InputRegion(nn.Module):
    """
    Contiguous region of MPS input cores, associated with bond_str = 'slri'
    """

    def __init__(
        self, tensor, use_bias=True, fixed_bias=True, bias_mat=None, ephemeral=False
    ):
        super().__init__()

        # Make sure tensor has correct size and the component mats are square
        assert len(tensor.shape) == 4
        assert tensor.size(1) == tensor.size(2)
        bond_dim = tensor.size(1)

        # If we are using bias matrices, set those up here
        if use_bias:
            assert bias_mat is None or isinstance(bias_mat, torch.Tensor)
            bias_mat = (
                torch.eye(bond_dim).unsqueeze(0) if bias_mat is None else bias_mat
            )

            bias_modes = len(list(bias_mat.shape))
            assert bias_modes in [2, 3]
            if bias_modes == 2:
                bias_mat = bias_mat.unsqueeze(0)

        # Register our tensors as a Pytorch Parameter or Tensor
        if ephemeral:
            self.register_buffer(name="tensor", tensor=tensor.contiguous())
            self.register_buffer(name="bias_mat", tensor=bias_mat)
        else:
            self.register_parameter(
                name="tensor", param=nn.Parameter(tensor.contiguous())
            )
            if fixed_bias:
                self.register_buffer(name="bias_mat", tensor=bias_mat)
            else:
                self.register_parameter(name="bias_mat", param=nn.Parameter(bias_mat))

        self.use_bias = use_bias
        self.fixed_bias = fixed_bias

    def forward(self, input_data):
        """
        Contract input with MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                                   feature_dim]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)

        # Contract the input with our core tensor
        mats = torch.einsum("slri,bsi->bslr", [tensor, input_data])

        # If we're using bias matrices, add those here
        if self.use_bias:
            bias_mat = self.bias_mat.unsqueeze(0)
            mats = mats + bias_mat.expand_as(mats)

        return MatRegion(mats)

    def _merge(self, offset):
        """
        Merge all pairs of neighboring cores and return a new list of cores

        offset is either 0 or 1, which gives the first core at which we start
        our merging. Depending on the length of our InputRegion, the output of
        merge may have 1, 2, or 3 entries, with the majority of sites ending in
        a MergedInput instance
        """
        assert offset in [0, 1]
        num_sites = self.core_len()
        parity = num_sites % 2

        # Cases with empty tensors might arise in recursion below
        if num_sites == 0:
            return [None]

        # Simplify the problem into one where offset=0 and num_sites is even
        if (offset, parity) == (1, 1):
            out_list = [self[0], self[1:]._merge(offset=0)[0]]
        elif (offset, parity) == (1, 0):
            out_list = [self[0], self[1:-1]._merge(offset=0)[0], self[-1]]
        elif (offset, parity) == (0, 1):
            out_list = [self[:-1]._merge(offset=0)[0], self[-1]]

        # The main case of interest, with no offset and an even number of sites
        else:
            tensor = self.tensor
            even_cores, odd_cores = tensor[0::2], tensor[1::2]
            assert len(even_cores) == len(odd_cores)

            # Multiply all pairs of cores, keeping inputs separate
            merged_cores = torch.einsum("slui,surj->slrij", [even_cores, odd_cores])
            out_list = [MergedInput(merged_cores)]

        # Remove empty MergedInputs, which appear in very small InputRegions
        return [x for x in out_list if x is not None]

    def __getitem__(self, key):
        """
        Returns an InputRegion instance sliced along the site index
        """
        assert isinstance(key, int) or isinstance(key, slice)

        if isinstance(key, slice):
            return InputRegion(self.tensor[key])
        else:
            return InputSite(self.tensor[key])

    def get_norm(self):
        """
        Returns list of the norms of each core in InputRegion
        """
        return [torch.norm(core) for core in self.tensor]

    @torch.no_grad()
    def rescale_norm(self, scale_list):
        """
        Rescales the norm of each core by an amount specified in scale_list

        For the i'th tensor defining a core in InputRegion, we rescale as
        tensor_i <- scale_i * tensor_i, where scale_i = scale_list[i]
        """
        assert len(scale_list) == len(self.tensor)

        for core, scale in zip(self.tensor, scale_list):
            core *= scale

    def core_len(self):
        return len(self)

    def __len__(self):
        return self.tensor.size(0)


class MergedInput(nn.Module):
    """
    Contiguous region of merged MPS cores, each taking in a pair of input data

    Since MergedInput arises after contracting together existing input cores,
    a merged input tensor is required for initialization
    """

    def __init__(self, tensor):
        # Check that our input tensor has the correct shape
        # bond_str = "slrij"
        shape = tensor.shape
        assert len(shape) == 5
        assert shape[1] == shape[2]
        assert shape[3] == shape[4]

        super().__init__()

        # Register our tensor as a Pytorch Parameter
        self.register_parameter(name="tensor", param=nn.Parameter(tensor.contiguous()))

    def forward(self, input_data):
        """
        Contract input with merged MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                 feature_dim], where input_dim must be even
                                 (each merged core takes 2 inputs)
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)
        assert input_data.size(1) % 2 == 0

        # Divide input_data into inputs living on even and on odd sites
        inputs = [input_data[:, 0::2], input_data[:, 1::2]]

        # Contract the odd (right-most) and even inputs with merged cores
        tensor = torch.einsum("slrij,bsj->bslri", [tensor, inputs[1]])
        mats = torch.einsum("bslri,bsi->bslr", [tensor, inputs[0]])

        return MatRegion(mats)

    def _unmerge(self, cutoff=1e-10):
        """
        Separate the cores in our MergedInput and return an InputRegion

        The length of the resultant InputRegion will be identical to our
        original MergedInput (same number of inputs), but its core_len will
        be doubled (twice as many individual cores)
        """
        # bond_str = "slrij"
        tensor = self.tensor
        svd_string = "lrij->lui,urj"
        max_D = tensor.size(1)

        # Split every one of the cores into two and add them both to core_list
        core_list, bond_list, sv_list = [], [None], [None]
        for merged_core in tensor:
            sv_vec = torch.empty(max_D)
            left_core, right_core, bond_dim = svd_flex(
                merged_core, svd_string, max_D, cutoff, sv_vec=sv_vec
            )

            core_list += [left_core, right_core]
            bond_list += [bond_dim, None]
            sv_list += [sv_vec, None]

        # Collate the split cores into one tensor and return as an InputRegion
        tensor = torch.stack(core_list)
        return [InputRegion(tensor)], bond_list, sv_list

    def get_norm(self):
        """
        Returns list of the norm of each core in MergedInput
        """
        return [torch.norm(core) for core in self.tensor]

    @torch.no_grad()
    def rescale_norm(self, scale_list):
        """
        Rescales the norm of each core by an amount specified in scale_list

        For the i'th tensor defining a core in MergedInput, we rescale as
        tensor_i <- scale_i * tensor_i, where scale_i = scale_list[i]
        """
        assert len(scale_list) == len(self.tensor)

        for core, scale in zip(self.tensor, scale_list):
            core *= scale

    def core_len(self):
        return len(self)

    def __len__(self):
        """
        Returns the number of input sites, which is twice the number of cores
        """
        return 2 * self.tensor.size(0)


class InputSite(nn.Module):
    """
    A single MPS core which takes in a single input datum, bond_str = 'lri'
    """

    def __init__(self, tensor):
        super().__init__()
        # Register our tensor as a Pytorch Parameter
        self.register_parameter(name="tensor", param=nn.Parameter(tensor.contiguous()))

    def forward(self, input_data):
        """
        Contract input with MPS core and return result as a SingleMat

        Args:
            input_data (Tensor): Input with shape [batch_size, feature_dim]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 2
        assert input_data.size(1) == tensor.size(2)

        # Contract the input with our core tensor
        mat = torch.einsum("lri,bi->blr", [tensor, input_data])

        return SingleMat(mat)

    def get_norm(self):
        """
        Returns the norm of our core tensor, wrapped as a singleton list
        """
        return [torch.norm(self.tensor)]

    @torch.no_grad()
    def rescale_norm(self, scale):
        """
        Rescales the norm of our core by a factor of input `scale`
        """
        if isinstance(scale, list):
            assert len(scale) == 1
            scale = scale[0]

        self.tensor *= scale

    def core_len(self):
        return 1

    def __len__(self):
        return 1


class OutputSite(nn.Module):
    """
    A single MPS core with no input and a single output index, bond_str = 'olr'
    """

    def __init__(self, tensor):
        super().__init__()
        # Register our tensor as a Pytorch Parameter
        self.register_parameter(name="tensor", param=nn.Parameter(tensor.contiguous()))

    def forward(self, input_data):
        """
        Return the OutputSite wrapped as an OutputCore contractable
        """
        return OutputCore(self.tensor)

    def get_norm(self):
        """
        Returns the norm of our core tensor, wrapped as a singleton list
        """
        return [torch.norm(self.tensor)]

    @torch.no_grad()
    def rescale_norm(self, scale):
        """
        Rescales the norm of our core by a factor of input `scale`
        """
        if isinstance(scale, list):
            assert len(scale) == 1
            scale = scale[0]

        self.tensor *= scale

    def core_len(self):
        return 1

    def __len__(self):
        return 0


class MergedOutput(nn.Module):
    """
    Merged MPS core taking in one input datum and returning an output vector

    Since MergedOutput arises after contracting together an existing input and
    output core, an already-merged tensor is required for initialization

    Args:
        tensor (Tensor):    Value that our merged core is initialized to
        left_output (bool): Specifies if the output core is on the left side of
                            the input core (True), or on the right (False)
    """

    def __init__(self, tensor, left_output):
        # Check that our input tensor has the correct shape
        # bond_str = "olri"
        assert len(tensor.shape) == 4
        super().__init__()

        # Register our tensor as a Pytorch Parameter
        self.register_parameter(name="tensor", param=nn.Parameter(tensor.contiguous()))
        self.left_output = left_output

    def forward(self, input_data):
        """
        Contract input with input index of core and return an OutputCore

        Args:
            input_data (Tensor): Input with shape [batch_size, feature_dim]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 2
        assert input_data.size(1) == tensor.size(3)

        # Contract the input with our core tensor
        tensor = torch.einsum("olri,bi->bolr", [tensor, input_data])

        return OutputCore(tensor)

    def _unmerge(self, cutoff=1e-10):
        """
        Split our MergedOutput into an OutputSite and an InputSite

        The non-zero entries of our tensors are dynamically sized according to
        the SVD cutoff, but will generally be padded with zeros to give the
        new index a regular size.
        """
        # bond_str = "olri"
        tensor = self.tensor
        left_output = self.left_output
        if left_output:
            svd_string = "olri->olu,uri"
            max_D = tensor.size(2)
            sv_vec = torch.empty(max_D)

            output_core, input_core, bond_dim = svd_flex(
                tensor, svd_string, max_D, cutoff, sv_vec=sv_vec
            )
            return (
                [OutputSite(output_core), InputSite(input_core)],
                [None, bond_dim, None],
                [None, sv_vec, None],
            )

        else:
            svd_string = "olri->our,lui"
            max_D = tensor.size(1)
            sv_vec = torch.empty(max_D)

            output_core, input_core, bond_dim = svd_flex(
                tensor, svd_string, max_D, cutoff, sv_vec=sv_vec
            )
            return (
                [InputSite(input_core), OutputSite(output_core)],
                [None, bond_dim, None],
                [None, sv_vec, None],
            )

    def get_norm(self):
        """
        Returns the norm of our core tensor, wrapped as a singleton list
        """
        return [torch.norm(self.tensor)]

    @torch.no_grad()
    def rescale_norm(self, scale):
        """
        Rescales the norm of our core by a factor of input `scale`
        """
        if isinstance(scale, list):
            assert len(scale) == 1
            scale = scale[0]

        self.tensor *= scale

    def core_len(self):
        return 2

    def __len__(self):
        return 1


class InitialVector(nn.Module):
    """
    Vector of ones and zeros to act as initial vector within the MPS

    By default the initial vector is chosen to be all ones, but if fill_dim is
    specified then only the first fill_dim entries are set to one, with the
    rest zero.

    If fixed_vec is False, then the initial vector will be registered as a
    trainable model parameter.
    """

    def __init__(self, bond_dim, fill_dim=None, fixed_vec=True, is_left_vec=True):
        super().__init__()

        vec = torch.ones(bond_dim)
        if fill_dim is not None:
            assert fill_dim >= 0 and fill_dim <= bond_dim
            vec[fill_dim:] = 0

        if fixed_vec:
            vec.requires_grad = False
            self.register_buffer(name="vec", tensor=vec)
        else:
            vec.requires_grad = True
            self.register_parameter(name="vec", param=nn.Parameter(vec))

        assert isinstance(is_left_vec, bool)
        self.is_left_vec = is_left_vec

    def forward(self):
        """
        Return our initial vector wrapped as an EdgeVec contractable
        """
        return EdgeVec(self.vec, self.is_left_vec)

    def core_len(self):
        return 1

    def __len__(self):
        return 0


class TerminalOutput(nn.Module):
    """
    Output matrix at end of chain to transmute virtual state into output vector

    By default, a fixed rectangular identity matrix with shape
    [bond_dim, output_dim] will be used as a state transducer. If fixed_mat is
    False, then the matrix will be registered as a trainable model parameter.
    """

    def __init__(self, bond_dim, output_dim, fixed_mat=False, is_left_mat=False):
        super().__init__()

        # I don't have a nice initialization scheme for a non-injective fixed
        # state transducer, so just throw an error if that's needed
        if fixed_mat and output_dim > bond_dim:
            raise ValueError(
                "With fixed_mat=True, TerminalOutput currently "
                "only supports initialization for bond_dim >= "
                "output_dim, but here bond_dim="
                f"{bond_dim} and output_dim={output_dim}"
            )

        # Initialize the matrix and register it appropriately
        mat = torch.eye(bond_dim, output_dim)
        if fixed_mat:
            mat.requires_grad = False
            self.register_buffer(name="mat", tensor=mat)
        else:
            # Add some noise to help with training
            mat = mat + torch.randn_like(mat) / bond_dim

            mat.requires_grad = True
            self.register_parameter(name="mat", param=nn.Parameter(mat))

        assert isinstance(is_left_mat, bool)
        self.is_left_mat = is_left_mat

    def forward(self):
        """
        Return our terminal matrix wrapped as an OutputMat contractable
        """
        return OutputMat(self.mat, self.is_left_mat)

    def core_len(self):
        return 1

    def __len__(self):
        return 0
