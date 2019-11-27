import json
import torch.utils.data as Data
import torch
from Models import Generator
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyDataSet(Data.Dataset):
    def __init__(self, datasets_filename, repeat=1):
        super(MyDataSet, self).__init__()
        with open(datasets_filename, 'r') as f:
            self.datasets = json.load(f)
        self.len = len(self.datasets)
        self.repeat = repeat

    def __getitem__(self, i):
        index = i % self.len
        # self.matrix = self.datasets[i]['matrix']
        # self.pin_locs = self.datasets['pin_locs']
        ret = self.datasets[i]['matrix']
        return torch.Tensor(ret)

    def __len__(self):
        if self.repeat is None:
            return 100000000
        else:
            return self.len * self.repeat


def visualization(adj, matrix):
    def plot_pins(grid, x, y):
        pass

    def plot_edge(grid):
        pass

    for i in range(adj.size(0)):
        adj_matrix = adj[i, :, :]
        m = matrix[i, :, :]


LR = 0.1
input_x = input_y = 20
conv_dims = [20, 50, 100, 50, 20]
BATCH_SIZE = 50
EPOCH = 10
ADJ_POW = 10
train_datasets_filename = r'try_datasets_train.json'
test_datasets_filename = r'try_datasets_test.json'

trainDataset = MyDataSet(train_datasets_filename, repeat=1)
testDataset = MyDataSet(test_datasets_filename, repeat=1)

adj_filter = torch.zeros((input_x * input_y, input_x * input_y))
for i in range(input_x * input_y):
    if i + 1 < input_x * input_y and (i + 1) % input_y != 0:
        adj_filter[i, i + 1] = adj_filter[i + 1, i] = 1
    if i + input_y < input_x * input_y:
        adj_filter[i, i + input_y] = adj_filter[i + input_y, i] = 1
adj_filter = adj_filter.tolist()
adj_filter = [adj_filter for _ in range(BATCH_SIZE)]
adj_filter = torch.Tensor(adj_filter)

trainDataLoader = Data.DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = Data.DataLoader(dataset=testDataset, batch_size=BATCH_SIZE, shuffle=False)

generatorNet = Generator(input_x=input_x, input_y=input_y, conv_dims=conv_dims, adj_filter=adj_filter).to(device)

optimizer = torch.optim.Adam(generatorNet.parameters(), LR)

opposite_identity = torch.ones((input_x * input_y, input_x * input_y))
for i in range(input_x * input_y):
    opposite_identity[i, i] = 0
opposite_identity = [opposite_identity.tolist() for _ in range(BATCH_SIZE)]
opposite_identity = torch.Tensor(opposite_identity)


def calculate_loss(matrix, adj):
    pin_vector = matrix[:, -1, :, :].view(BATCH_SIZE, -1, 1)  # change to column vector
    pin_matrix = torch.matmul(pin_vector, torch.transpose(pin_vector, 1, 2))
    adj_sum = sum(sum(sum(torch.matrix_power(adj, ADJ_POW) * pin_matrix * opposite_identity)))  # / 1e18

    length_matrix, width_matrix = matrix[:, 0, :, :], matrix[:, 1, :, :]
    area_vector = (length_matrix * width_matrix).view(BATCH_SIZE, 1, -1)

    wire_length = sum(sum(torch.matmul(torch.matmul(area_vector, adj), torch.transpose(area_vector, 1, 2))))
    # print(wire_length, adj_sum)

    # loss = -torch.sigmoid((adj_sum - 300 * wire_length) / 1e6)
    loss = -torch.sigmoid((adj_sum - 3000 * wire_length) / 1e9)
    # print(loss.shape)
    return loss


for epoch in range(EPOCH):
    for i, matrix in enumerate(trainDataLoader):
        optimizer.zero_grad()
        adj = generatorNet(matrix)
        loss = calculate_loss(matrix, adj)
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            for m in testDataLoader:
                adj_test = generatorNet(m)
                loss = calculate_loss(m, adj_test)
                print(loss)
                # visualization(adj_test)
