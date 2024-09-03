import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters:
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

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
        # Compute logits from token embeddings
        logits = self.token_embedding_table(idx) # (B, T, C)

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
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self.forward(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))