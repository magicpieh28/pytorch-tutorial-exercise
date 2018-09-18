import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

det, NN, v = "DET", "NN", "V"
text_data = [("The dog ate the apple".split(), [det, NN, v, det, NN]),
             ("Everybody read that book".split(), [NN, v, det, NN])]

def to_id(datum, dic: dict):
	for data in datum:
		if data not in dic:
			dic[data] = len(dic)
	return dic

def make_tensor(datum, dic: dict):
	inputs = [dic[data] for data in datum]
	return torch.tensor(inputs, dtype=torch.long)

vocab = {}
charset = {}
tagset = {}

for datum, tags in text_data:
	to_id(datum, vocab)
	to_id(tags, tagset)
	for data in datum:
		to_id(data, charset)

# vocab => {'The': 0, 'dog': 1, 'ate': 2, 'the': 3, 'apple': 4, 'Everybody': 5, 'read': 6,
#           'that': 7, 'book': 8}
# tagset => {'DET': 0, 'NN': 1, 'V': 2}
# charset => {'T': 0, 'h': 1, 'e': 2, 'd': 3, 'o': 4, 'g': 5, 'a': 6, 't': 7, 'p': 8, 'l': 9, 'E': 10,
#             'v': 11, 'r': 12, 'y': 13, 'b': 14, 'k': 15}

vocab_size = len(vocab)
charset_size = len(charset)
tagset_size = len(tagset)

c_embedding_dim = 32
c_hidden_dim = 32
w_embedding_dim = 32
hidden_dim = 32

class LSTMModel(nn.Module):
	def __init__(self, vocab_size, charset_size, tagset_size,
	             c_embedding_dim, c_hidden_dim, w_embedding_dim, hidden_dim):
		super().__init__()
		self.c_hidden_dim = c_hidden_dim
		self.hidden_dim = hidden_dim

		self.char_embedding = nn.Embedding(charset_size, c_embedding_dim)
		self.lstm1 = nn.LSTM(c_embedding_dim, c_hidden_dim)

		self.c_hx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))
		self.c_cx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))

		self.word_embedding = nn.Embedding(vocab_size, w_embedding_dim)
		self.lstm2 = nn.LSTM(w_embedding_dim + c_hidden_dim, hidden_dim)
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

		self.hx = nn.Parameter(torch.zeros(1, 1, hidden_dim))
		self.cx = nn.Parameter(torch.zeros(1, 1, hidden_dim))

		self.dropout = nn.Dropout()

	def forward(self, w_seq, c_seq):
		c_embeds = self.char_embedding(c_seq)
		char_lstm, _ = self.lstm1(c_embeds.view(len(c_seq), 1, -1), (self.c_hx, self.c_cx))

		w_embeds = self.word_embedding(w_seq)
		w_3d_embeds = w_embeds.view(len(w_seq), 1, -1)

		# print(torch.cat((char_lstm, w_3d_embeds), 0))

		lstm_out, _ = self.lstm2(torch.cat((char_lstm, w_3d_embeds), 0), (self.hx, self.cx))
		tag_space = self.hidden2tag(lstm_out.view(len(c_seq)+len(w_seq), -1))
		tag_score = F.log_softmax(tag_space, dim=1)
		return tag_score

model = LSTMModel(vocab_size, charset_size, tagset_size, c_embedding_dim,
                  c_hidden_dim, w_embedding_dim, hidden_dim)
optimizer = optim.SGD(model.parameters(), lr=0.05)
loss_func = nn.NLLLoss()

model.train()
for epoch in range(300):
	print(f'\n-- {epoch+1} --')
	for datum, tags in text_data:
		model.zero_grad()

		w_seq = make_tensor(datum, vocab)
		c_seq = torch.tensor([charset[i] for data in datum for i in data],
		                     dtype=torch.long)
		# for data in datum:
		# 	c_seq = make_tensor(data, charset)
		tag_score = model(w_seq, c_seq)

		target = make_tensor(tags, tagset)
		loss = loss_func(tag_score, target)
		loss.backward()
		optimizer.step()

		print(f'loss => {loss}')

