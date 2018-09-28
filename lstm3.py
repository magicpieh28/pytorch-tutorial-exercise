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

c_embedding_dim = 32
c_hidden_dim = 32
w_embedding_dim = 32
hidden_dim = 64
# RuntimeError: size mismatch, m1: [4 x 8], m2: [32 x 3] at
# RuntimeError: size mismatch, m1: [4 x 16], m2: [64 x 3] at

class LSTMModel(nn.Module):
	def __init__(self, vocab_size, charset_size, tagset_size,
	             c_embedding_dim, c_hidden_dim, w_embedding_dim, hidden_dim):
		super().__init__()
		self.c_hidden_dim = c_hidden_dim
		self.hidden_dim = hidden_dim

		self.char_embedding = nn.Embedding(charset_size, c_embedding_dim)
		# 17 * 32
		self.lstm1 = nn.LSTM(c_embedding_dim, c_hidden_dim)
		# 32 * 32

		self.c_hx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))
		self.c_cx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))

		self.word_embedding = nn.Embedding(vocab_size, w_embedding_dim)
		# 5 * 32
		self.lstm2 = nn.LSTM(w_embedding_dim + c_hidden_dim, hidden_dim)
		# (32 + 32) * 32

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

		self.hx = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
		self.cx = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

		self.dropout = nn.Dropout()

	def forward(self, w_seq, c_seq):
		print(w_seq, c_seq)
		c_embeds = self.char_embedding(c_seq)
		# torch.Size([3, 32])
		char_out, (c_hx, c_cx) = self.lstm1(c_embeds.view(len(c_seq), 1, -1),
		                                   (self.c_hx, self.c_cx))
		# char_out => torch.Size([3, 1, 32])

		w_embeds = self.word_embedding(w_seq)
		w_3d_embeds = w_embeds.view(len(w_seq), 1, -1)
		# torch.Size([1, 1, 32])

		# '그 결과인 최종 hidden state를 c_w로 하면 된다'!!! 아웃풋과 torch.cat()하는 것이 아니었어!
		lstm_out, _ = self.lstm2(torch.cat((c_hx, w_3d_embeds), 2),
		                         (self.hx, self.cx))
		# torch.cat()은 1, 1, 64가 되어있는데 lstm_out은 1, 1, 32가 되어 있어서
		# size mismatch 문제가 발생하는건가

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
		for idx, word in enumerate(datum):
			model.zero_grad()

			print(f'-- {idx} : {word} --')
			w_seq = torch.tensor([vocab[word]], dtype=torch.long)
			print(f'w_seq => {w_seq}')
			c_seq = torch.tensor([charset[char] for char in word], dtype=torch.long)
			print(f'c_seq => {c_seq}')

			tag_score = model(w_seq, c_seq)
			target = torch.tensor(tagset[tags[idx]], dtype=torch.long)
			print(f'target => {target}')

			loss = loss_func(tag_score, target)
			loss.backward()
			optimizer.step()

			print(f'loss => {loss}')