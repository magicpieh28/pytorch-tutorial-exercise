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
		# 17 * 32
		self.lstm1 = nn.LSTM(c_embedding_dim, c_hidden_dim)
		# 32 * 32

		self.c_hx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))
		self.c_cx = nn.Parameter(torch.zeros(1, 1, self.c_hidden_dim))

		self.word_embedding = nn.Embedding(vocab_size, w_embedding_dim)
		# 5 * 32
		self.lstm2 = nn.LSTM(w_embedding_dim + c_hidden_dim, hidden_dim)
		# (32 + 32) * 32
		# w_embedding_dim + c_hidden_dim은 32 + 32가 되어야 하는데 (그래서 64 에러)
		# char_lstm + w_embeds 를 0으로 torch.cat()하면 5 + 17이 되기 때문에 여전히 32
		# 그러면 torch.cat()할 때 0이 아니라 1로 해야 된다는 건가? 32쪽으로 더해야 하나?
		# 근데 1로는 더할수가 없는데 (5, 17이 안맞아서) 이걸 전치해서 더하면
		# torch.Size([32, 1, 17]) + torch.Size([32, 1, 5])라는건데 이건 더해서 전치하는 것과
		# 같지 않나? 결국에는 self.lstm2() 의 사이즈와는 크기의 위치?가 맞지 않아서 또 계산할 수 없음
		# 근데 튜토리얼 힌트에 w_embed랑 c_rep의 차원을 더한 것을 lstm2 인풋값으로 해야 한다고..
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

		self.hx = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
		self.cx = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

		self.dropout = nn.Dropout()

	def forward(self, w_seq, c_seq):
		print(w_seq, c_seq)
		c_embeds = self.char_embedding(c_seq)
		# torch.Size([17, 32])
		char_lstm, _ = self.lstm1(c_embeds.view(len(c_seq), 1, -1), (self.c_hx, self.c_cx))
		# torch.Size([17, 1, 32])

		w_embeds = self.word_embedding(w_seq)
		w_3d_embeds = w_embeds.view(len(w_seq), 1, -1)
		# torch.Size([5, 1, 32])

		# torch.cat((char_lstm, w_3d_embeds), 0)
		# torch.Size([22, 1, 32])

		lstm_out, _ = self.lstm2(torch.cat((char_lstm, w_3d_embeds), 2), (self.hx, self.cx))
		# cat()은 됐는데 self.lstm2()이 안돌아간다. 입력 사이즈가 64여야 하는데 32밖에 없단다.

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
		for word in datum:
			print(f'-- {word} --')
			w_seq = make_tensor(word, vocab)
			c_seq = torch.tensor([charset[char] for char in word], dtype=torch.long)

			model.zero_grad()
			target = make_tensor(tags, tagset)
			tag_score = model(w_seq, c_seq)

			loss = loss_func(tag_score, target)
			loss.backward()
			optimizer.step()

			print(f'loss => {loss}')