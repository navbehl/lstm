import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F

corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


tokenized_corpus = tokenize_corpus(corpus)
'''Vocabulary is basically a list of unique words with assigned indices.'''
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

'''We can now generate pairs center word, context word. Let’s assume context window to be
symmetric and equal to 2.'''
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, treated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

# it will be useful to have this as numpy array
idx_pairs = np.array(idx_pairs)

'''Input layer is just the center word encoded in one-hot manner. It dimensions are
 [1, vocabulary_size]'''


def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x


'''Hidden layer Hidden layer makes our v vectors. Therefore it has to have embedding_dims neurons.
 To compute it value we have to define W1 weight matrix. Of course its has to be
[embedding_dims, vocabulary_size]. There is no activation function — 
just plain matrix multiplication.'''

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims,
                          vocabulary_size).float(), requires_grad=True)
'''Output Layer.
Last layer must have vocabulary_size neurons — because it generates probabilities for each word.
 Therefore, W2 is [vocabulary_size, embedding_dims] in terms of shape.'''
W2 = Variable(torch.randn(vocabulary_size,
                          embedding_dims).float(), requires_grad=True)

num_epochs = 101
learning_rate = 0.001
for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        '''On top on that we have to use softmax layer'''
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1, -1), y_true)
        loss_val += loss.data[0]
        '''Backprop.'''
        loss.backward()
        '''SGD.'''
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        '''Last step is to zero gradients to make next pass clear.'''
        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

''' One, last thing is to extract vectors for words. It is possible in three ways:'''
print(W2)
