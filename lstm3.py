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

class LSTMModel(nn.Module):
	def __init__(self, vocab_size, charset_size, tagset_size,
	             c_embedding_dim, c_hidden_dim, w_embedding_dim, hidden_dim):
		super().__init__()
		self.c_hidden_dim = c_hidden_dim
		self.hidden_dim = hidden_dim

		self.c_hx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))
		self.c_cx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))

		self.char_embedding = nn.Embedding(charset_size, c_embedding_dim)
		self.lstm1 = nn.LSTM(c_embedding_dim, c_hidden_dim)

		self.word_embedding = nn.Embedding(vocab_size, w_embedding_dim)
		self.hx = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
		self.cx = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
		self.lstm2 = nn.LSTM(w_embedding_dim + c_hidden_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

		self.dropout = nn.Dropout()

	def forward(self, w_seq, c_seq):
		print(f'w_seq => {w_seq}')
		print(f'c_seq => {c_seq}')

		sentence_vec = []
		for word, char in zip(w_seq, c_seq):
			print(f'word => {word}')
			print(f'char => {char}')

			char_vec = self.char_embedding(char)
			char_vec = char_vec.view(char_vec.size(0), 1, char_vec.size(1))
			
			_, (c_hx, _) = self.lstm1(char_vec, (self.c_hx, self.c_cx))
			char_rep = c_hx[-1, ...]

			word_vec = self.word_embedding(word)

			vector = torch.cat((char_rep, word_vec), dim=1)
			sentence_vec.append(vector)

		sentence_vec = torch.stack(sentence_vec, dim=0)

		lstm_out, _ = self.lstm2(sentence_vec, (self.hx, self.cx))
		lstm_out = lstm_out.view(lstm_out.size(0), -1)

		tag_space = self.hidden2tag(lstm_out)
		tag_score = F.log_softmax(tag_space, dim=1)
		return tag_score

model = LSTMModel(vocab_size, charset_size, tagset_size,
                  c_embedding_dim=32, c_hidden_dim=32,
                  w_embedding_dim=32, hidden_dim=32)
optimizer = optim.SGD(model.parameters(), lr=0.05)
loss_func = nn.NLLLoss()

model.train()
for epoch in range(300):
	print(f'\n-- {epoch+1} --')
	for datum, tags in text_data:
		model.zero_grad()

		w_seq = [torch.tensor([vocab[data]], dtype=torch.long) for data in datum]
		c_seq = [torch.tensor([charset[d] for d in data], dtype=torch.long)\
		         for data in datum]

		tag_score = model(w_seq, c_seq)
		target = torch.tensor([tagset[tag] for tag in tags], dtype=torch.long)

		loss = loss_func(tag_score, target)
		loss.backward()
		optimizer.step()

		print(f'loss => {loss}')