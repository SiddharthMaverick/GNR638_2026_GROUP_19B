# Make sure the w64devkit is installed as in the repo
then make sure the source is set to this compiler

```bash
set PATH=%CD%\w64devkit\bin;%PATH%
```
then compile
```bash
g++ -O3 -shared -o libbackend.dll backend.cpp
```
then train using train.py
```bash
python train.py --data-path data_1 --epochs 100 --batch-size 32 --lr 0.01 --img-size 32 --classes 10 --seed 42 --save-dir ./checkpoints    
```
then test using script.py
```bash
python script.py --mode test --dataset data_1 --weights best_model_data_1.pkl --test-full
```



