pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
# python train.py --model ccn --batch_size 16  --checkpoint_path checkpoint-ccn
# python -m torch.distributed.launch train.py --model ccn --batch_size 16  --checkpoint_path checkpoint-ccn
# python  train.py --model ttn --batch_size 16 --checkpoint_path checkpoint-ttn-noxgf --lr 2e-4 --dist --print_interval 100 --save_interval 100
# python  train.py --model exp --batch_size 2 --checkpoint_path checkpoint-exp --lr 2e-4 --dist --print_interval 1 --save_interval 1 --early_stop --test_interval 1 --stop_interval 1
# python  train.py --model ccn --batch_size 16 --checkpoint_path checkpoint --lr 0.002 --print_interval 100 --save_interval 100 --dist
python  train.py --model ttn --batch_size 64 --checkpoint_path checkpoint --lr 2e-4 --print_interval 100 --save_interval 100 --dist