import torch


class Contractable:
    """
    Container for tensors with labeled indices and a global batch size

    The labels for our indices give some high-level knowledge of the tensor
    layout, and permit the contraction of pairs of indices in a more
    systematic manner. However, much of the actual heavy lifting is done
    through specific contraction routines in different subclasses

    Attributes:
        tensor (Tensor):    A Pytorch tensor whose first index is a batch
                            index. Sub-classes of Contractable may put other
                            restrictions on tensor
        bond_str (str):     A string whose letters each label a separate mode
                            of our tensor, and whose length equals the order
                            (number of modes) of our tensor
        global_bs (int):    The batch size associated with all Contractables.
                            This is shared between all Contractable instances
                            and allows for automatic expanding of tensors
    """

    # The global batch size
    global_bs = None

    def __init__(self, tensor, bond_str):
        shape = list(tensor.shape)
        num_dim = len(shape)
        str_len = len(bond_str)

        global_bs = Contractable.global_bs
        batch_dim = tensor.size(0)

        # Expand along a new batch dimension if needed
        if ("b" not in bond_str and str_len == num_dim) or (
            "b" == bond_str[0] and str_len == num_dim + 1
        ):
            if global_bs is not None:
                tensor = tensor.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError(
                    "No batch size given and no previous " "batch size set"
                )
            if bond_str[0] != "b":
                bond_str = "b" + bond_str

        # Check for correct formatting in bond_str
        elif bond_str[0] != "b" or str_len != num_dim:
            raise ValueError(
                "Length of bond string '{bond_str}' "
                f"({len(bond_str)}) must match order of "
                f"tensor ({len(shape)})"
            )

        # Set the global batch size if it is unset or needs to be updated
        elif global_bs is None or global_bs != batch_dim:
            Contractable.global_bs = batch_dim

        # Check that global batch size agrees with input tensor's first dim
        elif global_bs != batch_dim:
            raise RuntimeError(
                f"Batch size previously set to {global_bs}"
                ", but input tensor has batch size "
                f"{batch_dim}"
            )

        # Set the defining attributes of our Contractable
        self.tensor = tensor
        self.bond_str = bond_str

    def __mul__(self, contractable, rmul=False):
        """
        Multiply with another contractable along a linear index

        The default behavior is to multiply the 'r' index of this instance
        with the 'l' index of contractable, matching the batch ('b')
        index of both, and take the outer product of other indices.
        If rmul is True, contractable is instead multiplied on the right.
        """
        # This method works for general Core subclasses besides Scalar (no 'l'
        # and 'r' indices), composite contractables (no tensor attribute), and
        # MatRegion (multiplication isn't just simple index contraction)
        if (
            isinstance(contractable, Scalar)
            or not hasattr(contractable, "tensor")
            or type(contractable) is MatRegion
        ):
            return NotImplemented

        tensors = [self.tensor, contractable.tensor]
        bond_strs = [list(self.bond_str), list(contractable.bond_str)]
        lowercases = [chr(c) for c in range(ord("a"), ord("z") + 1)]

        # Reverse the order of tensors if needed
        if rmul:
            tensors = tensors[::-1]
            bond_strs = bond_strs[::-1]

        # Check that bond strings are in proper format
        for i, bs in enumerate(bond_strs):
            assert bs[0] == "b"
            assert len(set(bs)) == len(bs)
            assert all([c in lowercases for c in bs])
            assert (i == 0 and "r" in bs) or (i == 1 and "l" in bs)

        # Get used and free characters
        used_chars = set(bond_strs[0]).union(bond_strs[1])
        free_chars = [c for c in lowercases if c not in used_chars]

        # Rename overlapping indices in the bond strings (except 'b', 'l', 'r')
        specials = ["b", "l", "r"]
        for i, c in enumerate(bond_strs[1]):
            if c in bond_strs[0] and c not in specials:
                bond_strs[1][i] = free_chars.pop()

        # Combine right bond of left tensor and left bond of right tensor
        sum_char = free_chars.pop()
        bond_strs[0][bond_strs[0].index("r")] = sum_char
        bond_strs[1][bond_strs[1].index("l")] = sum_char
        specials.append(sum_char)

        # Build bond string of ouput tensor
        out_str = ["b"]
        for bs in bond_strs:
            out_str.extend([c for c in bs if c not in specials])
        out_str.append("l" if "l" in bond_strs[0] else "")
        out_str.append("r" if "r" in bond_strs[1] else "")

        # Build the einsum string for this operation
        bond_strs = ["".join(bs) for bs in bond_strs]
        out_str = "".join(out_str)
        ein_str = f"{bond_strs[0]},{bond_strs[1]}->{out_str}"

        # Contract along the linear dimension to get an output tensor
        out_tensor = torch.einsum(ein_str, [tensors[0], tensors[1]])

        # Return our output tensor wrapped in an appropriate class
        if out_str == "br":
            return EdgeVec(out_tensor, is_left_vec=True)
        elif out_str == "bl":
            return EdgeVec(out_tensor, is_left_vec=False)
        elif out_str == "blr":
            return SingleMat(out_tensor)
        elif out_str == "bolr":
            return OutputCore(out_tensor)
        else:
            return Contractable(out_tensor, out_str)

    def __rmul__(self, contractable):
        """
        Multiply with another contractable along a linear index
        """
        return self.__mul__(contractable, rmul=True)

    def reduce(self):
        """
        Return the contractable without any modification

        reduce() can be any method which returns a contractable. This is
        trivially possible for any contractable by returning itself
        """
        return self


class ContractableList(Contractable):
    """
    A list of contractables which can all be multiplied together in order

    Calling reduce on a ContractableList instance will first reduce every item
    to a linear contractable, and then contract everything together
    """

    def __init__(self, contractable_list):
        # Check that input list is nonempty and has contractables as entries
        if not isinstance(contractable_list, list) or contractable_list is []:
            raise ValueError("Input to ContractableList must be nonempty list")
        for i, item in enumerate(contractable_list):
            if not isinstance(item, Contractable):
                raise ValueError(
                    "Input items to ContractableList must be "
                    f"Contractable instances, but item {i} is not"
                )

        self.contractable_list = contractable_list

    def __mul__(self, contractable, rmul=False):
        """
        Multiply a contractable by everything in ContractableList in order
        """
        # The input cannot be a composite contractable
        assert hasattr(contractable, "tensor")
        output = contractable.tensor

        # Multiply by everything in ContractableList, in the correct order
        if rmul:
            for item in self.contractable_list:
                output = item * output
        else:
            for item in self.contractable_list[::-1]:
                output = output * item

        return output

    def __rmul__(self, contractable):
        """
        Multiply another contractable by everything in ContractableList
        """
        return self.__mul__(contractable, rmul=True)

    def reduce(self, parallel_eval=False):
        """
        Reduce all the contractables in list before multiplying them together
        """
        c_list = self.contractable_list
        # For parallel_eval, reduce all contractables in c_list
        if parallel_eval:
            c_list = [item.reduce() for item in c_list]

        # Multiply together all the contractables. This multiplies in right to
        # left order, but certain inefficient contractions are unsupported.
        # If we encounter an unsupported operation, then try multiplying from
        # the left end of the list instead
        while len(c_list) > 1:
            try:
                c_list[-2] = c_list[-2] * c_list[-1]
                del c_list[-1]
            except TypeError:
                c_list[1] = c_list[0] * c_list[1]
                del c_list[0]

        return c_list[0]


class MatRegion(Contractable):
    """
    A contiguous collection of matrices which are multiplied together

    The input tensor defining our MatRegion must have shape
    [batch_size, num_mats, D, D], or [num_mats, D, D] when the global batch
    size is already known
    """

    def __init__(self, mats):
        shape = list(mats.shape)
        if len(shape) not in [3, 4] or shape[-2] != shape[-1]:
            raise ValueError(
                "MatRegion tensors must have shape "
                "[batch_size, num_mats, D, D], or [num_mats,"
                " D, D] if batch size has already been set"
            )

        super().__init__(mats, bond_str="bslr")

    def __mul__(self, edge_vec, rmul=False):
        """
        Iteratively multiply an input vector with all matrices in MatRegion
        """
        # The input must be an instance of EdgeVec
        if not isinstance(edge_vec, EdgeVec):
            return NotImplemented

        mats = self.tensor
        num_mats = mats.size(1)

        # Load our vector and matrix batches
        dummy_ind = 1 if rmul else 2
        vec = edge_vec.tensor.unsqueeze(dummy_ind)
        mat_list = [mat.squeeze(1) for mat in torch.chunk(mats, num_mats, 1)]

        # Do the repeated matrix-vector multiplications in the proper order
        for i, mat in enumerate(mat_list[:: (1 if rmul else -1)], 1):
            if rmul:
                vec = torch.bmm(vec, mat)
            else:
                vec = torch.bmm(mat, vec)

        # Since we only have a single vector, wrap it as a EdgeVec
        return EdgeVec(vec.squeeze(dummy_ind), is_left_vec=rmul)

    def __rmul__(self, edge_vec):
        return self.__mul__(edge_vec, rmul=True)

    def reduce(self):
        """
        Multiplies together all matrices and returns resultant SingleMat

        This method uses iterated batch multiplication to evaluate the full
        matrix product in depth O( log(num_mats) )
        """
        mats = self.tensor
        shape = list(mats.shape)
        size, D = shape[1:3]

        # Iteratively multiply pairs of matrices until there is only one
        while size > 1:
            odd_size = size % 2 == 1
            half_size = size // 2
            nice_size = 2 * half_size

            even_mats = mats[:, 0:nice_size:2]
            odd_mats = mats[:, 1:nice_size:2]
            # For odd sizes, set aside one batch of matrices for the next round
            leftover = mats[:, nice_size:]

            # Multiply together all pairs of matrices (except leftovers)
            mats = torch.einsum("bslu,bsur->bslr", [even_mats, odd_mats])
            mats = torch.cat([mats, leftover], 1)

            size = half_size + int(odd_size)

        # Since we only have a single matrix, wrap it as a SingleMat
        return SingleMat(mats.squeeze(1))


class OutputCore(Contractable):
    """
    A single MPS core with a single output index
    """

    def __init__(self, tensor):
        # Check the input shape
        if len(tensor.shape) not in [3, 4]:
            raise ValueError(
                "OutputCore tensors must have shape [batch_size, "
                "output_dim, D_l, D_r], or else [output_dim, D_l,"
                " D_r] if batch size has already been set"
            )

        super().__init__(tensor, bond_str="bolr")


class SingleMat(Contractable):
    """
    A batch of matrices associated with a single location in our MPS
    """

    def __init__(self, mat):
        # Check the input shape
        if len(mat.shape) not in [2, 3]:
            raise ValueError(
                "SingleMat tensors must have shape [batch_size, "
                "D_l, D_r], or else [D_l, D_r] if batch size "
                "has already been set"
            )

        super().__init__(mat, bond_str="blr")


class OutputMat(Contractable):
    """
    An output core associated with an edge of our MPS
    """

    def __init__(self, mat, is_left_mat):
        # Check the input shape
        if len(mat.shape) not in [2, 3]:
            raise ValueError(
                "OutputMat tensors must have shape [batch_size, "
                "D, output_dim], or else [D, output_dim] if "
                "batch size has already been set"
            )

        # OutputMats on left edge will have a right-facing bond, and vice versa
        bond_str = "b" + ("r" if is_left_mat else "l") + "o"
        super().__init__(mat, bond_str=bond_str)

    def __mul__(self, edge_vec, rmul=False):
        """
        Multiply with an edge vector along the shared linear index
        """
        if not isinstance(edge_vec, EdgeVec):
            raise NotImplemented  # noqa: F901
        else:
            return super().__mul__(edge_vec, rmul)

    def __rmul__(self, edge_vec):
        return self.__mul__(edge_vec, rmul=True)


class EdgeVec(Contractable):
    """
    A batch of vectors associated with an edge of our MPS

    EdgeVec instances are always associated with an edge of an MPS, which
    requires the is_left_vec flag to be set to True (vector on left edge) or
    False (vector on right edge)
    """

    def __init__(self, vec, is_left_vec):
        # Check the input shape
        if len(vec.shape) not in [1, 2]:
            raise ValueError(
                "EdgeVec tensors must have shape "
                "[batch_size, D], or else [D] if batch size "
                "has already been set"
            )

        # EdgeVecs on left edge will have a right-facing bond, and vice versa
        bond_str = "b" + ("r" if is_left_vec else "l")
        super().__init__(vec, bond_str=bond_str)

    def __mul__(self, right_vec):
        """
        Take the inner product of our vector with another vector
        """
        # The input must be an instance of EdgeVec
        if not isinstance(right_vec, EdgeVec):
            return NotImplemented

        left_vec = self.tensor.unsqueeze(1)
        right_vec = right_vec.tensor.unsqueeze(2)
        batch_size = left_vec.size(0)

        # Do the batch inner product
        scalar = torch.bmm(left_vec, right_vec).view([batch_size])

        # Since we only have a single scalar, wrap it as a Scalar
        return Scalar(scalar)


class Scalar(Contractable):
    """
    A batch of scalars
    """

    def __init__(self, scalar):
        # Add dummy dimension if we have a torch scalar
        shape = list(scalar.shape)
        if shape is []:
            scalar = scalar.view([1])
            shape = [1]

        # Check the input shape
        if len(shape) != 1:
            raise ValueError(
                "input scalar must be a torch tensor with shape "
                "[batch_size], or [] or [1] if batch size has "
                "been set"
            )

        super().__init__(scalar, bond_str="b")

    def __mul__(self, contractable):
        """
        Multiply a contractable by our scalar and return the result
        """
        scalar = self.tensor
        tensor = contractable.tensor
        bond_str = contractable.bond_str

        ein_string = f"{bond_str},b->{bond_str}"
        out_tensor = torch.einsum(ein_string, [tensor, scalar])

        # Wrap the result in the same class right_contractable belongs to
        contract_class = type(contractable)
        if contract_class is not Contractable:
            return contract_class(out_tensor)
        else:
            return Contractable(out_tensor, bond_str)

    def __rmul__(self, contractable):
        # Scalar multiplication is commutative
        return self.__mul__(contractable)
