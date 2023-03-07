import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pprint import pprint

# Load the dataset into a pandas dataframe
# df = pd.read_csv("../data/train.csv")
DF = pd.read_csv("~/AI_WEEK/bug_detection/data/train.csv")

DEBUG = False
# model, case matters in code
# model = BertForSequenceClassification.from_pretrained('bert-base-cased')

# get the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

BS = 32

# get the max length of the sentences
# max_len = 0
# max_len_idx = 0
# for i, sent in enumerate(df['code']):
#     # split into 5 lines of code 
#     lines = sent.split('\n')
#     for each_line in lines:
#         if(len(each_line) > max_len):
#             max_len = len(each_line)
#             max_len_idx = i
#
# print(max_len, max_len_idx)

## dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = df['label'].to_numpy()
        # self.texts = [[tokenizer(loc,
        #                          padding='max_length', max_length = 512,
        #                          truncation='do_not_truncate',
        #                          return_tensors="pt")
        #                for loc in lines.split('\n')]
        #               for lines in df['code']]
        locs = [df['code'][idx].split('\n') for idx in df.index]
        third_lines = [loc[2] for loc in locs]
        # l, r = 0, 1000
        # self.third_lines = []
        tl_tokenized = tokenizer(third_lines,
                                 padding='max_length',
                                 max_length = 100,
                                 truncation=True,
                                 return_tensors="pt"
                               )
        self.third_lines = [dict(zip(tl_tokenized.data,t)) for t in zip(*tl_tokenized.data.values())]

        # doesnt work for some reason
        # while(r < len(third_lines)):
        #     tl_tokenized = tokenizer(third_lines[l:r],
        #                              padding='max_length',
        #                              max_length = 100,
        #                              truncation=True,
        #                              return_tensors="pt"
        #                             )
        #     l = r
        #     if r + 1000 < len(third_lines):
        #         r += 1000
        #     else:
        #         r = len(third_lines)
            # self.third_lines += [dict(zip(tl_tokenized.data,t)) for t in zip(*tl_tokenized.data.values())]


        # context_lines = [[loc[i] for i in range(len(loc)) if i != 2] for loc in locs]
        # context_lines = [list(x) for x in zip(*context_lines)]
        # self.context_lines = [[]] * len(context_lines)
        # for i in range(4):
        #     cl_tokenized = tokenizer(context_lines[i],
        #                      padding='max_length',
        #                      max_length = 100,
        #                      truncation=True,
        #                      return_tensors="pt"
        #                     )
        #     self.context_lines[i] += [dict(zip(cl_tokenized.data,t)) for t in zip(*cl_tokenized.data.values())]
        # self.context_lines = [list(x) for x in zip(*self.context_lines)]
            

        context_lines = ["".join([loc[i] for i in range(len(loc)) if i != 2]) for loc in locs]
        cl_tokenized = tokenizer(context_lines,
                                 padding='max_length',
                                 max_length = 150,
                                 truncation=True,
                                 return_tensors="pt"
                               )
        self.context_lines = [dict(zip(cl_tokenized.data,t)) for t in zip(*cl_tokenized.data.values())]
            # l, r = 0, 1000
            # while(r < len(context_lines[i])):
            #     cl_tokenized = tokenizer(context_lines[i][l:r],
            #                      padding='max_length',
            #                      max_length = 100,
            #                      truncation=True,
            #                      return_tensors="pt"
            #                     )
            #     l = r
            #     if r + 1000 < len(context_lines[i]):
            #         r += 1000
            #     else:
            #         r = len(context_lines[i])
            #     self.context_lines[i] += [dict(zip(cl_tokenized.data,t)) for t in zip(*cl_tokenized.data.values())]

        # self.context_lines = [[dict(zip(cl,t)) for t in zip(*cl.values())] for cl in cl_tokenized]


        extensions = {
        "c" : 0,
        "cpp" : 1,
        "java" : 2,
        "js" : 3,
        "kt" : 4,
        "py" : 5,
        "rs" : 6,
        "ts" : 7
        }
        self.extensions = [extensions[df['file_extension'][idx]] for idx in df.index]
        # for idx in df.index:
        #     loc = df['code'][idx].split('\n')
        #     self.third_lines.append(tokenizer(loc[2], 
        #                                  padding='max_length',
        #                                  max_length = 512,
        #                                  truncation='do_not_truncate',
        #                                  return_tensors="pt"))
        #     self.context_lines.append([tokenizer(line, 
        #                                  padding='max_length',
        #                                  max_length = 512,
        #                                  truncation='do_not_truncate',
        #                                  return_tensors="pt") 
        #                                  for i, line in enumerate(loc) if i != 2])
            # self.extensions.append(extensions[df['file_extension'][idx]])

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # if idx >= len(self.labels):
        #     raise Exception(f"Index {idx} is out of bounds")
        return self.labels[idx]

    def get_batch_codes(self, idx):
        # if idx >= len(self.labels):
        #     raise Exception(f"Index {idx} is out of bounds")
        return self.third_lines[idx], self.context_lines[idx], torch.tensor(self.extensions[idx])

    def __getitem__(self, idx):
        # if idx >= len(self.labels):
        #     raise Exception(f"Index {idx} is out of bounds")
        batch_x = self.get_batch_codes(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_x, batch_y


rng = np.random.default_rng(seed=42)


df_train, df_val, df_test = np.split(DF.sample(frac=1, random_state=42), 
                                     [int(.8*len(DF)), int(.9*len(DF))])

print(len(df_train),len(df_val), len(df_test))


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bert_third_lines = BertModel.from_pretrained('bert-base-uncased')
        self.bert_context_lines = BertModel.from_pretrained('bert-base-uncased')
        # bruh this shit is gonna take 6 hours on a 3080ti
        self.third_line_nn = torch.nn.Sequential(
                torch.nn.Linear(768, 768//8),
                torch.nn.LayerNorm(768//8),
                torch.nn.ReLU(),
                torch.nn.Linear(768//8, 768//8),
                torch.nn.LayerNorm(768//8),
                torch.nn.ReLU(),
                torch.nn.Linear(768//8, 768//16),
                torch.nn.LayerNorm(768//16),
                torch.nn.ReLU(),
            )
        self.context_lines_nn = torch.nn.Sequential(
                # torch.nn.Linear(768 * 4, 768),
                # torch.nn.LayerNorm(768),
                # torch.nn.ReLU(),
                torch.nn.Linear(768, 768//8),
                torch.nn.LayerNorm(768//8),
                torch.nn.ReLU(),
                torch.nn.Linear(768//8, 768//8),
                torch.nn.LayerNorm(768//8),
                torch.nn.ReLU(),
                torch.nn.Linear(768//8, 768//16),
                torch.nn.LayerNorm(768//16),
                torch.nn.ReLU(),
            )
        self.final_nn = torch.nn.Sequential(
                # torch.nn.Linear(768 * 2 + 1, 768),
                torch.nn.Linear(768//8 + 1, 768//16),
                torch.nn.LayerNorm(768//16),
                torch.nn.ReLU(),
                torch.nn.Linear(768//16, 768//32),
                torch.nn.LayerNorm(768//32),
                torch.nn.ReLU(),
                torch.nn.Linear(768//32, 768//64),
                torch.nn.LayerNorm(768//64),
                torch.nn.ReLU(),
                torch.nn.Linear(768//64, 768//128),
                torch.nn.LayerNorm(768//128),
                torch.nn.ReLU(),
                torch.nn.Linear(768//128, 1),
                # torch.nn.Linear(768 * 2 + 1, 1),
                torch.nn.Sigmoid()
            )


    def forward(self, feature_vec):
        # _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        third_line_id = feature_vec[0]["input_ids"].to(self.device)
        third_line_mask = feature_vec[0]["attention_mask"].to(self.device)
        _, third_line_pooled_output = self.bert_third_lines(input_ids = third_line_id, attention_mask=third_line_mask, return_dict = False)


        # print(third_line_pooled_output.size())

        # context_lines = zip([line["input_ids"].to(self.device) for line in feature_vec[1]], [line["attention_mask"].to(self.device) for line in feature_vec[1]])
        #
        # # context_lines_pooled_output = torch.tensor([self.bert(input_ids = in_id, attention_mask=mask, return_dict = False)[1] for in_id, mask in context_lines])
        # context_lines_pooled_output = torch.tensor([]).to(self.device)
        # for in_id, mask in context_lines:
        #     context_lines_pooled_output = torch.cat((context_lines_pooled_output, self.bert(input_ids = in_id, attention_mask=mask)["pooler_output"]), dim=1) 

        # print(context_lines_pooled_output.size())

        context_lines_id = feature_vec[1]["input_ids"].to(self.device)
        context_lines_mask = feature_vec[1]["attention_mask"].to(self.device)
        _, context_lines_pooled_output = self.bert_context_lines(input_ids = context_lines_id, attention_mask=context_lines_mask, return_dict = False)

        extensions = torch.reshape(feature_vec[2], (feature_vec[2].size()[0], 1)).to(self.device)
                
        # not waiting six fucking hours
        # return self.final_nn(torch.cat((self.third_line_nn(third_line_pooled_output), 
        #               self.context_lines_nn(context_lines_pooled_output),
        #               extensions), dim=1)).reshape((BS,))
        return self.final_nn(torch.cat((self.third_line_nn(third_line_pooled_output), 
                      self.context_lines_nn(context_lines_pooled_output),
                      extensions), dim=1))# .reshape((feature_vec[2].size()[0],))

        # layer_norm_output = self.layer_norm(pooled_output)
        # linear_output = self.linear(layer_norm_output)
        # final_layer = self.relu(linear_output)
        #
        # return final_layer

    def set_device(self, device):
        self.device = device

def train(model, train_data, val_data, learning_rate, epochs):
    
    print("Loading datasets (tokenizing)")

    train, val = Dataset(train_data), Dataset(val_data)

    print("Now the torch loader")

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BS)

    print("Loading datasets done...")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.set_device(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    
    print("loss and optimizer chosen...")
    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
            print("using cuda...")

    for epoch_num in range(epochs):
            # print(f"Starting Epoch #{epoch_num}")
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                # pprint(train_input)
                # return

                train_label = train_label.to(device)
                # mask = train_input['attention_mask'].to(device)
                # 
                # input_id = train_input['input_ids'].squeeze(1).to(device)
                #
                output = model(train_input)
                # breakpoint()
                
                # print(train_label.float())
                # print(output)
                # breakpoint()
                batch_loss = criterion(output, train_label.float().unsqueeze(1))
                total_loss_train += batch_loss.item()
                
                acc = (output.round() == train_label.unsqueeze(1)).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    # mask = val_input['attention_mask'].to(device)
                    # input_id = val_input['input_ids'].squeeze(1).to(device)
                    #
                    # output = model(input_id, mask)
                    output = model(val_input)

                    batch_loss = criterion(output, val_label.float().unsqueeze(1))
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.round() == val_label.unsqueeze(1)).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  
EPOCHS = 3
model = Classifier()
LR = 1e-6
              
train(model, df_train if not DEBUG else df_train.sample(10000), df_val if not DEBUG else df_val.sample(10000), LR, EPOCHS)

def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size = BS)

    model = model.cuda()

    total_acc_test = 0
    output_labels = []
    output_probs = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(torch.device('cuda'))
            output = model(test_input)
            output_labels += output.round().reshape(-1).to(torch.int8).tolist()
            output_probs += output.reshape(-1).tolist()
            acc = (output.round() == test_label.unsqueeze(1)).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return output_probs, output_labels

x = evaluate(model, df_test if not DEBUG else df_test.sample(10000))
print(f"max probability was {max(x[0])}, number of true values was {sum(x[1])}, number of false values was {len(x[1]) - sum(x[1])}")

with open('model_state3.pt', 'wb') as f:
    torch.save(model.state_dict(), f)
