from trainer import HaliteTrainer
from model import VBM
from nn import wings_model
import time

mz,msh,msy = wings_model()
vbm = VBM(mz,msh,msy)

trainer = HaliteTrainer(vbm,batch_size=1000)

for j in range(200):
    for i in range(400):
        trainer.step()
    trainer.fit()
    trainer.reset()
