module load pytorch/1.12.1
source activate epi
salloc -p devel -N 1 -n 1 -c 20 --gres=gpu:1 --time=00:20:00
srun --jobid=579312 --pty bash

# salloc --account=su114-gpu -p gpu-devel -N 1 -n 1 -c 1 --mem-per-cpu=3860 --gres=gpu:ampere_a100:3 --time=1:00:00
python train.py /jmain02/home/J2AD015/axf03/zxl79-axf03/repository/evodiff/config/config38M.json /jmain02/home/J2AD015/axf03/zxl79-axf03/repository/evodiff/dataset --dataset=/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/evodiff/dataset/sequence.csv --warmup
