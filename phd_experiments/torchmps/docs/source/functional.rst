Functional MPS
##############

The possible arguments for defining an MPS are:

 * `input_dim`: The dimension of the input we feed to our MPS
 * `output_dim`: The dimension of the output we get from each input
 * `bond_dim`: The internal bond dimension, a hyperparameter that sets the expressivity of our MPS. When in adaptive training mode, `bond_dim` instead sets the **maximum** possible bond dimension, with the initial bond dimension equal to `output_dim`
 * `feature_dim`: The dimension of the local feature spaces we embed each datum in (_default = 2_)
 * `adaptive_mode`: Whether our MPS is trained with its bond dimensions chosen adaptively or are fixed at creation (_default = False (fixed bonds)_)
 * `periodic_bc`: Whether our MPS has periodic boundary conditions (making it a tensor ring) or open boundary conditions (_default = False (open boundaries)_)
 * `parallel_eval`: For open boundary conditions, whether contraction of tensors is performed serially or in parallel (_default = False (serial)_)
 * `label_site`: The location in the MPS chain where our output lives after contracting all other sites with inputs (_default = input_dim // 2_)
 * `path`: A list specifying the path our MPS takes through the input data. For example, `path = [0, 1, ..., input_dim-1]` gives the standard in-order traversal (used if `path = None`), while `path = [0, 2, ..., input_dim-1]` defines an MPS which only acts on even-valued sites within our input (_default = None_)
 * `cutoff`: The singular value cutoff which controls adaptation of bond dimensions (_default = 1e-9_)
 * `merge_threshold`: The number of inputs before our MPS dynamically shifts its merge state, which updates half the bond dimensions at a time (_default = 2000, only used in adaptive mode_)
 * `init_std`: The size of the random terms used during initialization (_default = 1e-9_)

To define a custom feature map for embedding input data, first define a function `feature_map` which acts on a single scalar input and outputs a Pytorch vector of size `feature_dim`. After initializing an MPS `my_mps`, simply call `my_mps.register_feature_map(feature_map)`, and the user-specified `feature_map` will be applied to all input data given to `my_mps`. If `feature_map` is also a Pytorch Module, then any parameters associated with the map will be included in `my_mps.parameters()`. This streamlines the use of trainable feature maps within an MPS model.
