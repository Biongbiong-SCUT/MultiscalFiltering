import numpy as np 
import matplotlib.pyplot as plt 
without_init_shrink = [1960, 1960, 1960, 1960, 1960]
without_int_8 = [1960, 1632, 1481, 1447, 1428]
without_init_16 = [1960, 1473, 1375, 1353, 1341] # lr = 16
without_init_32 = [1960, 1305, 1244, 1230, 1222]
with_init = [388, 388, 388, 388, 388]
with_init_1 = [388, 339, 306, 277, 258] # lr = 1
with_init_16 = [388, 238, 205, 193, 187] # lr = 16
with_init_8 = [388, 260, 224, 208, 199]
with_init_32 = [388, 216, 189, 181, 178]


plt.plot(without_init_shrink, '--v',label="without init and shrink ")
plt.plot(without_int_8, '-v', label="without init shrink rate = 8")
plt.plot(without_init_16, '-v', label="without init shrink rate = 16")
plt.plot(without_init_32, '-v', label="without init shrink rate = 32")

plt.plot(with_init, '--o',label="with init and shrink")
plt.plot(with_init_1, '-o',label="with init shrink rate = 1")
plt.plot(with_init_8, '-o',label="with init shrink rate = 8")
plt.plot(with_init_16, '-o',label="with init shrink rate = 16")
plt.plot(with_init_32, '-o',label="with init shrink rate = 32")
plt.xlabel("k Iterative Number.")
plt.ylabel("Average Neighbor Facet Num.")
plt.legend()
plt.show()