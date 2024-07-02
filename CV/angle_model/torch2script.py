import torch

path = r"C:\Code\ML\Model\angle_model\angle_in64_model01.pt"
model = torch.load(path)
model.eval()
model = torch.jit.script(model)
torch.jit.save(model, r"C:\Code\ML\Model\angle_model\script_angle_in64_model01.pt")