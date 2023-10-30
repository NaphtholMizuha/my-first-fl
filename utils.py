import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


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
    w_packed = torch.Tensor([]).to('cuda')
    for key in table.keys():
        # flatten the weight tensors and contact them into one vector
        w_packed = torch.cat((w_packed, torch.flatten(w[key])))
    # padding the vector so that the length of the vector is multiple of `n`
    pad = n - (w_packed.size()[0] % n)
    w_packed = torch.cat((w_packed, torch.zeros(size=[pad]).to('cuda')))
    w_packed = w_packed.view(n, -1)
    return w_packed

def unpack(w, table):
    """
    Recover the weight dict from chunks
    :param w: weight chunks
    :param table: table which notes the keys and weight tensors' shape of layers in NN model
    :return: unpacked weight state_dict
    """
    w_dict = {}
    w = w.view(-1)
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
    dct = dct_mat(n)[:m].to('cuda')
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
    padded = torch.zeros(n, w.size()[1]).to('cuda')
    padded[:m] = w
    idct = dct_mat(n, inverse=True).to('cuda')
    reconstructed = idct.matmul(padded)
    return reconstructed

def differential_privacy(w, n_clients, n_comm, threshold, epsilon=1, delta=0.1):
    std = 2 * threshold * np.sqrt(n_comm * (-np.log(delta))) / (epsilon * n_clients)
    w /= max(1, torch.norm(w) / threshold)
    w += torch.Tensor(np.random.normal(scale=std, size=w.shape)).to('cuda')
    return w

def plot_contrast(title, data, labels):
    colors = ['red', 'blue', 'green', 'violet', 'orange', 'brown']
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for i, datum in enumerate(data):
        plt.plot(datum, color=colors[i], label=labels[i])
        plt.legend()
        plt.title(title)
    plt.savefig(f'./{title}.png')
    plt.close()

def dataset_partition(labels, alpha, n_client, type="iid"):
    labels = torch.Tensor(labels)
    n_class = int(labels.max() + 1)
    if type == "iid":
        rand_perm = np.random.permutation(len(labels))
        num_cumsum = np.cumsum(np.full(n_client, len(labels) / n_client)).astype(int)
        client_indices_pairs = [(cid, idxs) for cid, idxs in
                                enumerate(np.split(rand_perm, num_cumsum)[:-1])]
        return dict(client_indices_pairs)

    elif type == "dirichlet":
        label_dist = np.random.dirichlet(np.full(n_client, alpha), n_class)
        class_indices = [np.argwhere(labels == y).flatten()
                    for y in range(n_class)]

        client_idcs = {i : np.array([]).astype(int) for i in range(n_client)}
        for k_idcs, fracs in zip(class_indices, label_dist):
            for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
                idcs = idcs.numpy().astype(int)
                client_idcs[i] = np.append(client_idcs[i], idcs)

        return client_idcs

def show_distribution(dataset, dist_dict, n_client, n_class):
    mat = np.zeros([n_client, n_class])
    for client, indices in dist_dict.items():
        for idx in indices:
            mat[client][int(dataset.targets[idx])] += 1
    print(mat)
    plt.matshow(mat, vmin=0, vmax=4000, cmap=plt.cm.Blues)
    plt.xlabel('Client')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Class')
    plt.title("IID distribution")
    plt.colorbar()
    plt.savefig('./dist.png')
    plt.close()