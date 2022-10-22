import os
import numpy as np
survey1 = 9627
while os.path.exists('./data/'+str(survey1)):
    print("cover")
    survey1 = np.random.randint(10000)
os.mkdir('./data/'+str(survey1))
