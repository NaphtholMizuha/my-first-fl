import numpy as np
import torch


def dct_mat(n, inverse=False):
    """
    Generate a DCT matrix with type torch.Tensor
    :param n: size of the matrix
    :return: DCT matrix
    """
    ans = np.empty([n, n])
    ans[0, :] = 1 * np.sqrt(1 / n)
    for i in range(1, n):
        for j in range(n):
            ans[i, j] = np.sqrt(2 / n) * np.cos(np.pi * i * (2 * j + 1) / (2 * n))
    ans = torch.from_numpy(ans).float()
    if inverse:
        return torch.t(ans)
    else:
        return ans

def gen_pack_table(w: dict):
    table = {}
    start = 0
    for key, value in w.items():
        shape = value.shape
        length = value.view(-1).shape[0]
        table[key] = (start, length, shape)
        start += length
    return table
def pack(w, n, table):
    """
    Pack the state_dict into many chunks with shape n*1
    :param w: origin weight_dict
    :param n: size of chunks
    :param table: table which notes the names of all layers
    :return: packed weight (shape: n * number_of_chunks)
    """
    w_packed = torch.Tensor([])
    for key in table.keys():
        # flatten the weight tensors and contact them into one vector
        w_packed = torch.cat((w_packed, torch.flatten(w[key])))
    # padding the vector so that the length of the vector is multiple of `n`
    pad = n - (w_packed.size()[0] % n)
    w_packed = torch.cat((w_packed, torch.zeros(size=[pad])))
    return w_packed.view(n, -1)

def unpack(w, table):
    """
    Recover the weight dict from chunks
    :param w: weight chunks
    :param table: table which notes the keys and weight tensors' shape of layers in NN model
    :return: unpacked weight state_dict
    """
    w_dict = {}

    for key in table.keys():
        first, length, shape = table[key]
        w_dict[key] = w[first: first + length].view(shape)

    return w_dict

def compress(w, n, m):
    """
    compress the chunks with (partial) DCT matrix
    :param w: packed weight chunks
    :param n: size of original chunks
    :param m: size of compressed chunks
    :return: compressed packed weight (shape: m * number_of_chunks)
    """
    dct = dct_mat(n)[:m]
    compressed = dct.matmul(w)
    return compressed.view(-1)

def reconstruct(w, n, m):
    """
    reconstruct the chunks with IDCT matrix
    :param w: compressed weight chunks
    :param n: size of original chunks
    :param m: size of compressed chunks
    :return: reconstruct weight chunks
    """
    w = w.view(m, -1)
    padded = torch.zeros(n, w.size()[1])
    padded[:m] = w
    idct = dct_mat(n, inverse=True)
    reconstructed = idct.matmul(padded).view(-1)
    return reconstructed

def differential_privacy(w, threshold, std):
    w /= torch.max(1, torch.norm(w, dim=1) / threshold)
    w += torch.randn(w.shape) * std
    return w