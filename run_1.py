import numpy as np
from src.ADW import ADW
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.random.randint(0,20, 150)
    y = np.random.randint(0,20, 150)
    z = np.random.randint(15,37,150)
    extent = [0,20,0,20]
    
    obj = ADW(x,y,z,extent,res=0.25)
    obj.interpolate()
    
    lon = obj.xgrd
    lat = obj.ygrd
    estimado = obj.grid
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot()
    c = ax.contourf(lon,lat,estimado,level=[25,26,27,28,29,30])
    plt.colorbar(c)
    ax.scatter(x,y,color='red',label='Points')
    ax.legend()
    plt.savefig('sample.png')