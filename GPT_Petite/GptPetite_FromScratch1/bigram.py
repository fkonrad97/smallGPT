# https://arxiv.org/abs/1706.03762 - Attention is all you need!!

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters:
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for prediction?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # https://pytorch.org/docs/stable/notes/mps.html
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2   # Dropout randomly blocks some of the neurons from communicate from each other: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
# ------------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encode the text to indices
decode = lambda l: ''.join([itos[i] for i in l]) # decode the index list to text

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix]) # stack function puts the one dimensional tensor into a multi-dimensional tensor
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # it will not call backward on this, we do not intend to backpropagate on this
def estimate_loss():
    out = {}
    model.eval() # Setting the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # setting the model to training mode
    return out    

# Scale Dot-Product attention - from paper
class Head(nn.Module):
    ''' one head of self-attention '''
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))    # 'tril' is not a parameter of the module, so we have to assign it with the 'register_buffer' to it
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)      # (B,T,C)
        q = self.query(x)    # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) = (B,T,T) * head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v   # (B,T,T) @ (B,T,C) = (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    ''' multiple heads of self-attention in parallel '''
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # for residual connections: deep neural network optimization
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # the multiplication by 4 is because of: '3.3. Position-wise Feed-Forward Networks' -> the inner layer in the feedforward should be 4 times than the input and the output = adding more computation
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # instead of having a self.proj again after the Sequential
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # residual connections: to add to itself -> deep nn optimazation
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # self-position embedding table
        # Replaced by self.blocks:
        # self.sa_head = Head(n_embd) - single head self-attention
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention, because of the torch.cat, it should match the initial size
        # self.ffwd = FeedForward(n_embd)
        '''
        The * operator before the list comprehension in the code self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) is used to unpack the list of Block objects.
        The * operator is used to unpack the list. This means that instead of passing the list itself as a single argument to nn.Sequential, it passes each element of the list as a separate argument.
        '''
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])  # This equals below to the 'blocks', only this is more flexible, not hardcoded
        self.ln_f = nn.LayerNorm(n_embd) # final layernorm
        # self.blocks = nn.Sequential(
        #    Block(n_embd, n_head=4),
        #    Block(n_embd, n_head=4),
        #    Block(n_embd, n_head=4),
        #    nn.LayerNorm(n_embd),
        #)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass of the language model.
        
        Args:
            idx: Tensor of shape (B, T) containing token indices.
            targets: Tensor of shape (B, T) containing target token indices, or None.
        
        Returns:
            logits: Tensor of shape (B, T, C) containing the raw, unnormalized scores for each token.
            loss: Computed loss value if targets are provided, else None.
        """
        B, T = idx.shape
        
        # Compute logits from token embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) - now x has the information from not only the token embeddings, rather the position embedding as well
        # Replaced by self.blocks:
        # x = self.sa_heads(x) # apply one head of self-attention. (B,T,C)
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T, vocab_size)

        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            ''' https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss:
                    The description of the crossEntropy function states the it needs its inputs in a way that the,
                    'C' - channels are in the 2nd dimension, so that is why we need to reshape our tensors
            '''
            '''
                logits tensor has shape (B, T, C) after the forward pass, 
                where B is the batch size, 
                T is the sequence length, 
                and C is the number of classes (vocab size)
            '''
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T) # We could use -1 as well
            loss = F.cross_entropy(logits, targets)
    
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens from the model.
        
        Args:
            idx: Tensor of shape (B, T) containing the current context indices.
            max_new_tokens: Number of new tokens to generate.
        
        Returns:
            idx: Tensor of shape (B, T + max_new_tokens) containing the extended sequence.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, because of: nn.Embedding(block_size, n_embd)
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
print('Device: ', device)
m = model.to(device)

# Create a PyTorch optimizer for train the model: (We used to use SGD optimizer, but Adam is more efficient)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zero out the gradients
    loss.backward() # calculate the new gradients
    optimizer.step() # update the gradients

# generate fro the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# m.generate(context, max_new_tokens=500)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))