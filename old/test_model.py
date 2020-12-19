import threading
from model import ServerModel
import time

def print_fn(ltxt):
    #print("-",ltxt)
    return ltxt

def test_server_model1():
    t = time.perf_counter()
    sm = ServerModel(print_fn)
    def th_fn(v):
        for i in range(5):
            assert v == sm.request(v)
    ths = []
    for i in range(100):
        th = threading.Thread(target=th_fn,args=(i,))
        ths.append(th)
    for th in ths:
        th.start()
    print("passed : ",time.perf_counter()-t,'sec')

if __name__ == '__main__':
    test_server_model1()
