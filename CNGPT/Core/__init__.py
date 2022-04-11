import torch
print("===============================================")
print("GPTCore Load Done!!")
print("Device :{}".format(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
print("PyTorch version :{}".format(torch.__version__))
print("===============================================)
