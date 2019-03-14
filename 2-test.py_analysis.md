## test.py

------

process:

![1551768287108](img/1551768287108.png)

![1551765465051](img/1551765465051.png)

| parameter name |                      meaning                      |
| :------------: | :-----------------------------------------------: |
|     config     |              Configuration file name              |
|   checkpoint   |                  checkpoint file                  |
|      gpus      |                   Number of gpu                   |
|  proc_per_gpu  | The number of processes per gpu, the default is 1 |
|      out       |                name of output file                |
|      eval      |                Authentication type                |
|      show      |                    show result                    |

Parsing the configuration file:

![1551765577983](img/1551765577983.png)

Load datasets:

![1551765620987](img/1551765620987.png)

Model is created and data is loaded

​    1.Single gpu

![1551765661119](img/1551765661119.png)

2. Multiple gpus

   ![1551765690878](img/1551765690878.png)