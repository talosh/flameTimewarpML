import torch
import matplotlib.pyplot as plt

def inverse_exponential_torch(x, k=1):
    print ('hello from inverse exp')
    return (1 - torch.exp(-k * x))
    # return torch.sqrt(x) / 2

def exponential_torch(x, k=1):
    return torch.exp(k * x)

def custom_bend(x):
    linear_part = x
    l = 4
    k = 0.25
    # exp_positive = 1 + (l * (1 - torch.exp( - (k/torch.sqrt(x)) * (x - 1))))
    # exp_positive = 1 + (l * (1 - torch.exp( - (k) * (x - 1))))
    exp_positive = torch.pow(x, 1/3)
    exp_negative = -torch.pow(-x, 1/3)
    # exp_negative = -1 - (l * (1 - torch.exp( (k) * (x + 1))))
    return torch.where(x > 1, exp_positive, torch.where(x < -1, exp_negative, linear_part))

def custom_de_bend(x, l=1, k=1):
    linear_part = x
    inv_positive = torch.pow(x, 3)
    inv_negative = -torch.pow(-x, 3)
    # inv_positive = 1 + torch.log(1 - (x - 1) / 4) / -( k / x )
    # inv_negative = -1 + torch.log(1 + (x + 1) / 4) / k
    return torch.where(x > 1, inv_positive, torch.where(x < -1, inv_negative, linear_part))

x = torch.linspace(-200, 200, 400).to(device=torch.device('mps'), dtype = torch.float32)
m = custom_bend(x, l=4, k=0.25)
m = torch.tanh(m)
m = torch.arctanh(m)
y = custom_de_bend(m, l=4, k=0.25)

# y = custom_bend(x, l=4, k=0.25)

plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Inverse Exponential Function in PyTorch')
plt.grid(True)
plt.show()
