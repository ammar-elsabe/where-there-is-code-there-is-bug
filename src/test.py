import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pprint import pprint

df = pd.read_csv("~/AI_WEEK/bug_detection/data/test.csv")


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

BS = 42

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        locs = [df['code'][idx].split('\n') for idx in df.index]
        third_lines = [loc[2] for loc in locs]
        tl_tokenized = tokenizer(third_lines,
                                 padding='max_length',
                                 max_length = 100,
                                 truncation=True,
                                 return_tensors="pt"
                               )
        self.third_lines = [dict(zip(tl_tokenized.data,t)) for t in zip(*tl_tokenized.data.values())]

            

        context_lines = ["".join([loc[i] for i in range(len(loc)) if i != 2]) for loc in locs]
        cl_tokenized = tokenizer(context_lines,
                                 padding='max_length',
                                 max_length = 175,
                                 truncation=True,
                                 return_tensors="pt"
                               )
        self.context_lines = [dict(zip(cl_tokenized.data,t)) for t in zip(*cl_tokenized.data.values())]

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

    # def classes(self):
    #     return self.labels

    def __len__(self):
        # return len(self.labels)
        return len(self.third_lines)

    # def get_batch_labels(self, idx):
    #     # if idx >= len(self.labels):
    #     #     raise Exception(f"Index {idx} is out of bounds")
    #     return self.labels[idx]

    def get_batch_codes(self, idx):
        # if idx >= len(self.labels):
        #     raise Exception(f"Index {idx} is out of bounds")
        return self.third_lines[idx], self.context_lines[idx], torch.tensor(self.extensions[idx])

    def __getitem__(self, idx):
        # if idx >= len(self.labels):
        #     raise Exception(f"Index {idx} is out of bounds")
        batch_x = self.get_batch_codes(idx)
        # batch_y = self.get_batch_labels(idx)

        return batch_x#, batch_y



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
                torch.nn.Linear(768//8, 768//4),
                torch.nn.LayerNorm(768//4),
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
                torch.nn.Linear(768//8, 768//4),
                torch.nn.LayerNorm(768//4),
                torch.nn.ReLU(),
            )
        self.final_nn = torch.nn.Sequential(
                # torch.nn.Linear(768 * 2 + 1, 768),
                torch.nn.Linear(768//2 + 1, 768),
                torch.nn.LayerNorm(768),
                torch.nn.ReLU(),
                torch.nn.Linear(768, 768//8),
                torch.nn.LayerNorm(768//8),
                torch.nn.ReLU(),
                torch.nn.Linear(768//8, 768//8),
                torch.nn.LayerNorm(768//8),
                torch.nn.ReLU(),
                torch.nn.Linear(768//8, 768//8),
                torch.nn.LayerNorm(768//8),
                torch.nn.ReLU(),
                torch.nn.Linear(768//8, 1),
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


with open("model_state_final_04.pt", 'rb') as f:
    model = Classifier()
    model.load_state_dict(torch.load(f))
    model.set_device(torch.device('cuda'))
test = Dataset(df)

test_dataloader = torch.utils.data.DataLoader(test, batch_size = BS)

model = model.cuda()

labels = []
with torch.no_grad():
    for test_input in tqdm(test_dataloader):
        output = model(test_input)
        labels += output.round().reshape(-1).to(torch.int8).tolist()

output_frame = pd.DataFrame(df.pop('id'))
output_frame['label'] = labels
output_frame.to_csv("final_submission.csv", index=False)
