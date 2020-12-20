from trainer import HaliteTrainer
from model import VBM
from nn import wings_model
import time

mz,msh,msy = wings_model()
vbm = VBM(mz,msh,msy)

trainer = HaliteTrainer(vbm,batch_size=1000)

t = time.perf_counter()
trainer.step()
print(time.perf_counter()-t)
