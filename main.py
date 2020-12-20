from trainer import HaliteTrainer
from model import VBM
from nn import wings_model
import time

mz,msh,msy = wings_model()
vbm = VBM(mz,msh,msy)

trainer = HaliteTrainer(vbm,batch_size=100)
t = time.perf_counter()
for i in range(40):
    t2 = time.perf_counter()
    trainer.step()
    print(time.perf_counter()-t2)
t2 = time.perf_counter()
trainer.reset()
print(time.perf_counter()-t2)
print(time.perf_counter()-t)
