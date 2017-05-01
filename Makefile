cifar:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_gbrbm_cifar.py

.PHONY: clean
clean:
	rm -rf *.pyc rbm_plots/ models/
