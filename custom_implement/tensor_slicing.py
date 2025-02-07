import torch

def slicing(inputs, n_dimension, idx):
    # Slice index for the corresponding dimension.
    slice_indices = [idx,idx+1]

    # Create a tuple containing the slicing indices.
    slicing_indices = [slice(None)] * len(inputs.shape)
    slicing_indices[n_dimension] = slice(*slice_indices)

    # Perform slicing operations.
    sliced_tensor = inputs[tuple(slicing_indices)]
    sliced_tensor = sliced_tensor.squeeze(n_dimension)
    return sliced_tensor

if __name__ == '__main__':
    # Generate a tensor with variable dimensions.
    tensor = torch.arange(2*2*3*6*6).reshape(2,2,3,6,6)

    sl = slicing(tensor, 2, 0) # == tensor[:,:,0], extracts index 0 from the third dimension.
    print(sl)