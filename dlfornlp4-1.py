import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

raw_text = """We are about to study the idea of a computational process/
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

context_size = 4
embedding_dim = 10

vocab = set(raw_text)
vocab_size = len(vocab)
print(vocab_size)

word_to_id = {word : id for id, word in enumerate(vocab)}

data = []
for i in range(2, len(raw_text)-2):
	context = [raw_text[i-2], raw_text[i-1],
	           raw_text[i+1], raw_text[i+2]]Å
	target = raw_text[i]
	data.append((context, target))
print(data)

class CBOW(nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, context_to_id):
		embed = self.embedding(context_to_id).view((1, -1))
		# print(f'before view => {self.embedding(context_to_id).size()}')
		sembed = sum(embed).view((1, -1))
		out = F.relu(self.linear1(sembed))
		# print(f'out1 => {out}')
		out = self.linear2(out)
		log_probs = - F.log_softmax(out, dim=1)
		return log_probs

loss_func = nn.NLLLoss()
model = CBOW(vocab_size, embedding_dim, context_size)
optimizer = optim.SGD(model.parameters(), lr=0.001)
losses = []

def make_context_vec(context, word_to_id):
	indice = [word_to_id[word] for word in context]
	return torch.tensor(indice, dtype=torch.long)

for epoch in range(10):
	total_loss = torch.Tensor([0])
	for context, target in data:
		context_to_id = make_context_vec(context, word_to_id)
		# print(f'context_to_id => {context_to_id.size(), context_to_id}')
		model.zero_grad()
		log_probs = model(context_to_id)
		loss = loss_func(log_probs, torch.tensor([word_to_id[target]], dtype=torch.long))
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	losses.append(total_loss)

print(f'losses => {losses}')