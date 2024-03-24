import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import os
import json
import re
import pickle
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import model as custom_model

data_path = './'
seq2seqmodel_path = "./seq2seqModel.h5"
i2w_pickle_file = './i2w.pickle'

class DataProcessor(Dataset):
    def __init__(self, label_file, files_dir, dictionary, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.avi = files_reader(label_file)
        self.w2i = w2i
        self.dictionary = dictionary
        self.data_pair = process_data(files_dir, dictionary, w2i)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000
        return torch.Tensor(data), torch.Tensor(sentence)

class test_data_loader(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)

    def __getitem__(self, idx):
        return self.avi[idx]

def create_dictionary(word_min):
    with open(data_path + 'training_label.json', 'r') as f:
        file = json.load(f)
    wc = {}
    for d in file:
        for s in d['caption']:
            ws = re.sub('[.!,;?]]', ' ', s).split()
            for word in ws:
                word = word.replace('.', '') if '.' in word else word
                if word in wc:
                    wc[word] += 1
                else:
                    wc[word] = 1

    dict_1 = {}
    for word in wc:
        if wc[word] > word_min:
            dict_1[word] = wc[word]

    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(dict_1)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(dict_1)}

    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index

    return i2w, w2i, dict_1

def split_sentence(sentence, dictionary, w2i):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in dictionary:
            sentence[i] = 3
        else:
            sentence[i] = w2i[sentence[i]]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence

def process_data(label_file, dictionary, w2i):
    label_json = label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = split_sentence(s, dictionary, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption

def word_to_index(w2i, w):
    return w2i[w]

def index_to_word(i2w, i):
    return i2w[i]

def sentence_to_indices(w2i, sentence):
    return [w2i[w] for w in sentence]

def indices_to_words(i2w, index_seq):
    return [i2w[int(i)] for i in index_seq]

def files_reader(files_dir):
    avi_data = {}
    training_feats = files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

def train(model, epoch, train_loader, loss_func):
    model.train()
    print(f'Epoch: {epoch}')
    model = model
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = avi_feats, ground_truths
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)

        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(seq_logProb, ground_truths, lengths, loss_func)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 10 == 9:
            print(f"Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/10:.3f}")
            running_loss = 0.0

    average_loss = running_loss / len(train_loader)
    print(f'Epoch: {epoch}, Average Loss: {average_loss:.3f}')
    scheduler.step(average_loss)

def evaluate(test_loader, model):
    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        val1, val2, lengths = batch
        val1, val2 = val1, val2
        val1, val2 = Variable(val1), Variable(val2)
        seq_logProb, seq_predictions = model(val1, mode='inference')
        val2 = val2[:, 1:]
        test_predictions = seq_predictions[:3]
        test_truth = val2[:3]
        break

def test_function(test_loader, model, i2w):
    model.eval()
    ss = []
    for batch_idx, batch in enumerate(test_loader):
        identifier, f1 = batch
        identifier, f1 = identifier, Variable(f1).float()
        seq_logProb, seq_predictions = model(f1, mode='inference')
        test_predictions = seq_predictions
        res = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        res = [' '.join(s).split('<EOS>')[0] for s in res]

        rr = zip(identifier, res)
        for r in rr:
            ss.append(r)
    return ss

def calculate_loss(x, y, lengths, loss_fn):
    bs = len(x)
    p_cat = None
    g_cat = None
    flag = True
    for batch in range(bs):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1
        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            p_cat = predict
            g_cat = ground_truth
            flag = False
        else:
            p_cat = torch.cat((p_cat, predict), dim=0)
            g_cat = torch.cat((g_cat, ground_truth), dim=0)
    loss = loss_fn(p_cat, g_cat)
    avg_loss = loss / bs
    return loss

def create_mini_batch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

def main():
    label_file = data_path + 'testing_data/feat'
    files_dir = data_path + 'testing_label.json'
    test_dataset = DataProcessor(label_file, files_dir, dictionary, w2i)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=create_mini_batch)
   
    epochs_n = 20
    ModelSaveLoc = (seq2seqmodel_path)
    with open(i2w_pickle_file, 'wb') as f:
         pickle.dump(i2w, f)

    x = len(i2w) + 4
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)
    loss_fn = nn.CrossEntropyLoss()
    encoder_net = custom_model.EncoderNet()
    decoder_net = custom_model.DecoderNet(512, x, x, 1024, 0.3)
    model_train = custom_model.ModelMain(encoder=encoder_net, decoder=decoder_net) 

    start = time.time()
    for epoch in range(epochs_n):
        train(model_train, epoch + 1, train_loader=test_dataloader, loss_func=loss_fn)
        evaluate(test_dataloader, model_train)

    end = time.time()
    torch.save(model_train, "{}/{}.h5".format(ModelSaveLoc, 'seq2seqModel'))
    print("Training finished {} elapsed time: {:.3f} seconds.\n".format('test', end - start))

if __name__ == "__main__":
    main()
