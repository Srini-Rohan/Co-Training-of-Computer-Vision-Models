import torch

device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")


class dequeue():
  
  def __init__(self,maxlen):
    self.maxlen = maxlen
    self.values = []

  def add(self,value):
    self.values.append(value)
    if(len(self.values)>self.maxlen):
      self.values.pop(0)
      
  def average(self):
    return sum(self.values)/self.maxlen




def accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model([x,{1:1,2:0,3:1,4:0,5:1,6:1,7:1,8:1,9:1,10:1,11:0,12:1,13:0,14:1,15:0,16:1},1])
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples