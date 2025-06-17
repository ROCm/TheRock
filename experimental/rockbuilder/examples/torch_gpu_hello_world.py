import torch

print("Pytorch version: " + torch.__version__)
print("ROCM HIP version: " + torch.version.hip)
X_train = torch.FloatTensor([0.0, 1.0, 2.0])
print("cuda device count: " + str(torch.cuda.device_count()))
print("default cuda device name: " + torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device type: " + str(device))
X_train = X_train.to(device)
print("Tensor training running on cuda: " + str(X_train.is_cuda))
print("Running simple model training test")
print("    " + str(X_train))
print("Hello World, test executed succesfully")
