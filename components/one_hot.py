def one_hot(tensor, out_dim):
    y_onehot = tensor.new(*tensor.shape[:-1], out_dim).zero_()
    y_onehot.scatter_(-1, tensor.long(), 1)
    return y_onehot.float()
