import pickle
import numpy as np
import glob



for file in glob.glob("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/TCGA_Square_Imgs/Metastatic_data/SquareImg/5_dim_images/*.dat"):
    with open(file, 'rb') as f:
        x = pickle.load(f)
        f.close()
        id = f.name.split("/")[10]

        for i in range(5):
            before = x[i, : , :]
            tmp = np.array(x[i, : , :])
            np.random.shuffle(tmp.ravel())
            tmp = tmp.reshape((198,198))
            x[i, :, :] = tmp
    with open('/home/mateo/pytorch_docker/TCGA_GenomeImage/data/TCGA_Square_Imgs/Metastatic_data/Shuffle/5_dim_images/{}'.format(id), 'wb') as out:
        pickle.dump(x, out, protocol=pickle.HIGHEST_PROTOCOL)
print(tmp)