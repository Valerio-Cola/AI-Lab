# Model class must be defined somewhere
model = torch.load(PATH, weights_only=False)
model.eval()