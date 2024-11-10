"""
This code is for detecting anomaly parts in images using a Convolutional Autoencoder (CAE).
It can be extended to use other models if specified, like AE or AAE.

Make sure to specify the folder that stores the Bowtie dataset, as well as the model you intend to use (CAE, AE, etc.).

Example:
    Train:
        python train.py --data_dir [dataset folder] --model [model_name]
"""

from options.train_options import TrainOptions
from data import create_dataset  # Updated import for dataset
from models import create_model
import time
from utils.utils import plt_show


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # get dataset (bowtie)
    dataset_size = len(dataset)
    print(f"Training size is = {dataset_size}")

    model = create_model(opt)  # create model (e.g., CAE, AE)
    model.setup(opt)  # set model: if mode is 'train', define schedulers; if 'test', load saved networks
    total_iters = 0
    loss_name = model.loss_name  # loss name for naming

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # start epoch time
        model.update_learning_rate(epoch)  # update learning rate with schedulers
        epoch_iters = 0

        for i, data in enumerate(dataset):  # dataset loop
            iter_start_time = time.time()  # start iter time
            model.set_input(data)  # unpack input data for processing
            model.train()  # train model
            total_iters += 1
            epoch_iters += 1

        # Print loss and timing information at specified frequency
        if epoch % opt.print_epoch_freq == 0:
            losses = model.get_current_losses(*loss_name)
            epoch_time = time.time() - epoch_start_time
            message = f"epoch : {epoch} | total_iters : {total_iters} | epoch_time:{epoch_time:.3f}"
            for k, v in losses.items():
                message += f" | {k}:{v}"
            print(message)

        # Save model at specified frequency
        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
            model.save_networks()
            plt_show(model.generated_imgs[:3])  # Show sample generated images
