import numpy as np
from datasets.data_feeder import ensure_divisible

def main():
    # a = np.random.randint(10, size=(9, 5))
    # print(a)
    # wl = 3
    # ws = 2
    # wo = 1
    # b = np.array(list(zip(*(a[i::ws][:] for i in range(wl)))))
    # print(b)

    length, nf = ensure_divisible(5119, 400, 160)
    print(length)
    print(nf)


if __name__=="__main__":
    main()