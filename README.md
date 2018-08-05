# graph steady state embedding
Learning Steady-States of Iterative Algorithms over Graphs (coming soon)

#### 1. Setup the environment

##### 1) Download the repository

    git clone git@github.com:Hanjun-Dai/steady_state_embedding --recursive
    
##### 2) Build the dependency

This project depends on the graphnn library. The building instruction can be found here:

    https://github.com/Hanjun-Dai/graphnn
    
##### 3) Download the data
Use the following dropbox link:

    https://www.dropbox.com/sh/3gwr2wgh455q9pi/AAB0i6EQimVGslrqtsTIWsL0a?dl=0

Put everything under the 'data' folder, or create a symbolic link with name 'data':

    ln -s /path/to/your/downloaded/files data

Finally the folder structure should look like this:

    steady_state_embedding (project root)
    |__  README.md
    |__  code
    |__  graphnn
    |__  data
    |__  |__ algo_data
    |    |__ amazon
    |    |__ pagerank_ba
    |......
    
#### 2. Run the experiments

Most of the experiments are self-contained, where you need to build the code and then execute 
the script. For example:

    cd code/fit_algo/connectivity
    make
    ./connect.sh
    
The data is already cooked, so the ``code/data_process`` folder is for reference purpose. 

#### Reference

```bibtex
@inproceedings{dai2018learning,
  title={Learning Steady-States of Iterative Algorithms over Graphs},
  author={Dai, Hanjun and Kozareva, Zornitsa and Dai, Bo and Smola, Alex and Song, Le},
  booktitle={International Conference on Machine Learning},
  pages={1114--1122},
  year={2018}
}
```
