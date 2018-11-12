import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
# import torchvision
import numpy as np
from utils.word2vec import load_embedding
import pandas as pd


w2vmodel, embedding_matrix = load_embedding()
root_path = './data/'


def word2idx(word):
    """Return index for the word."""
    return w2vmodel.wv.vocab[word].index


def idx2word(idx):
    """Return word for the index."""
    return w2vmodel.wv.index2word[idx]


def sentences_to_indices(x, max_len):
    """Map sentences to indices."""
    m = x.shape[0]
    x_indices = np.zeros((m, max_len))
    print(x[0])
    for i in range(m):
        sentence_words = x[i].lower().split()
        j = 0
        for w in sentence_words:
            if w in w2vmodel.wv:
                x_indices[i, j] = word2idx(w)
                j = j + 1
    return x_indices


class AdrDataset(Data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if self.train:
            dataset = pd.read_csv(
                './data/adr classification data.csv', encoding='ISO-8859-1'
            ).groupby('Clinical ADR text').first().reset_index()
            traindata = dataset.iloc[:, 0].values
            trainlabels = dataset.iloc[:, 2].values
            self.train_data = sentences_to_indices(
                traindata, 10)
            self.train_labels = trainlabels
            # self.train_labels = convert_to_one_hot(trainlabels, C=5)
        else:
            pass

    def __getitem__(self, index):
        if self.train:
            data, target = self.train_data[index], self.train_labels[index]
        else:
            pass
        if self.transform is not None:
            pass
        if self.target_transform is not None:
            pass
        return data, target

    def __len__(self):
        if self.train:
            return 180
        else:
            return 0


train_data = AdrDataset(
    root=root_path,
    train=True,
    # transform=torchvision.transforms.ToTensor(),
)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=20, shuffle=True, num_workers=0)


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        weights = torch.FloatTensor(w2vmodel.syn0)
        self.word_embeds = nn.Embedding.from_pretrained(weights)
        self.rnn = nn.LSTM(embedding_matrix.shape[1], 128, 2,
                           batch_first=True, dropout=0.5)
        self.linear = nn.Linear(128, 5)
        self.out = nn.Softmax()

    def forward(self, x, h):
        out = self.word_embeds(x)
        out, _ = self.rnn(out, h)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.out(out)
        return out


model = MyModel()
# model.word_embeds.parameters().requires_grad  = False

for param in model.parameters():
    param.requires_grad = False
    break

model = model.cuda()

loss_func = nn.CrossEntropyLoss()

optimizer1 = torch.optim.Adam(model.rnn.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model.linear.parameters(), lr=0.001)

for epoch in range(50):
    for step, (data, target) in enumerate(train_loader):
        data = data.long()
        target = target.long()
        model.zero_grad()
        states = (Variable(torch.zeros(2, 20, 128)).cuda(),
                  Variable(torch.zeros(2, 20, 128)).cuda())
        input = Variable(data).cuda()
        target = Variable(target).cuda()
        output = model(input, states)
        loss = loss_func(output, target)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    if (epoch + 1) % 10 == 0:
        x_test = np.array(['not feeling happy', 'Holy shit',
                           'you are so pretty', 'let us play ball'])
        X_test_indices = sentences_to_indices(x_test, 10)
        X_test_indices = torch.from_numpy(X_test_indices)
        X_test_indices = Variable(X_test_indices.long()).cuda()
        states = (Variable(torch.zeros(2, 4, 128)).cuda(),
                  Variable(torch.zeros(2, 4, 128)).cuda())
        pred = model(X_test_indices, states)
        for i in range(len(x_test)):
            num = np.argmax(pred.data[i])
            print(' prediction: ' + x_test[i] + num)

torch.save(model.state_dict(), "./data/adr_classification.pkl")
