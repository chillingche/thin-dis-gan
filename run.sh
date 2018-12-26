if [ "$1" != "" ];then
    python $1 --batch-size=128 --lr=2e-4 --niter=46900 --cuda --ngpu=1
    # python $1 --batch-size=512 --lr=2e-4 --niter=50 --cuda --ngpu=1 --root=/data/dataset/cifar
    # python $1 --batch-size=512 --lr=2e-4 --niter=50 --cuda --ngpu=2
    # python $1 --batch-size=64 --lr=1e-4 --niter=50 --cuda --ngpu=2
    # python $1 --batch-size=64 --lr=1e-4 --niter=50 --cuda --ngpu=1
    # python $1 --batch-size=512 --lr=1e-3 --niter=50 --cuda --ngpu=1
    # python $1 --batch-size=512 --lr=2e-3 --niter=50 --cuda --ngpu=1
    # python $1 --batch-size=1024 --lr=2e-3 --niter=50 --cuda --ngpu=2
fi
