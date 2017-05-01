mkdir data && cd data

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --no-check-certificate
tar -zxvf cifar-10-python.tar.gz

rm cifar-10-python.tar.gz && cd ..
