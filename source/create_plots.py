from dataloading import data_load
from autoencoder_fully import AutoEncoder
from noisy import add_multiplicative, add_gaussian, add_noise

from matplotlib import pyplot as plt
import seaborn as sns
import torch



if __name__ == "__main__":
    BATCH = 32
    VAL_SIZE = 10000

    test_id = 38

    train_loader, val_loader, test_loader, _ = data_load(validation_size=VAL_SIZE,
                                                        batch_size=BATCH, 
                                                        visualize_split=False)
        
    device = "cuda"

    # load best model for testing
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(f'/home/akahige/Python Work/Denoising/archive/model_ckpts/fully_cnn_{test_id}/{test_id}.pt'))
    model.eval()

    id_ = 18

    for batch, labels in test_loader:
        noisy = add_noise(batch).to(device)
        output = model(noisy)
        rgb = batch.permute((0, 2, 3, 1)).cpu()
        output = output.permute((0, 2, 3, 1)).detach().cpu()
        noisy = noisy.permute((0, 2, 3, 1)).cpu()
        
        fig, ax = plt.subplots(1, 3)
        
        ax[0].imshow(rgb[id_])
        ax[0].set_title("Initial RGB Image")
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        
        ax[1].imshow(noisy[id_])
        ax[1].set_title("Noisy Image")
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        
        ax[2].imshow(output[id_])
        ax[2].set_title("Model Output")
        ax[2].set_yticklabels([])
        ax[2].set_xticklabels([])
        ax[2].set_yticks([])
        ax[2].set_xticks([])
        
        fig.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(1, 3, sharey=True)

        ax[0].set_title("Initial RGB Image")
        sns.kdeplot(rgb[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[0], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(rgb[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[0], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(rgb[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[0], color='blue', label='Blue Channel').set(xlim=(0, 1))

        ax[0].legend()

        
        ax[1].set_title(f"Noisy Image")
        sns.kdeplot(noisy[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[1], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(noisy[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[1], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(noisy[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[1], color='blue', label='Blue Channel').set(xlim=(0, 1))
            
        ax[1].legend()


        ax[2].set_title(f"Model Output")
        sns.kdeplot(output[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[2], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(output[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[2], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(output[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[2], color='blue', label='Blue Channel').set(xlim=(0, 1))

        ax[2].legend()
        
        plt.suptitle("Color Distribution")
        fig.text(0.5, 0.04, 'Normalized Color Value', ha='center')

        plt.show()
        
        noise_type_1 = add_gaussian(batch, mean=0.3, sigma=0.05)
        
        noise_type_1 = noise_type_1.permute((0, 2, 3, 1)).cpu()
        
        noise_type_2 = add_multiplicative(batch, mean=0.8, sigma=0.5)
        
        noise_type_2 = noise_type_2.permute((0, 2, 3, 1)).cpu()
        
        noise_type_1_2 = add_gaussian(batch, mean=0.2, sigma=0.05)
        noise_type_1_2 = add_multiplicative(noise_type_1_2, mean=0.2, sigma=0.1)
        
        noise_type_1_2 = noise_type_1_2.permute((0, 2, 3, 1)).cpu()
        
        noise_type_2_1 = add_multiplicative(batch, mean=0.2, sigma=0.2)
        noise_type_2_1 = add_gaussian(noise_type_2_1, mean=0.1, sigma=0.08)
        
        noise_type_2_1 = noise_type_2_1.permute((0, 2, 3, 1)).cpu()
        
        fig, ax = plt.subplots(1, 5)
        
        ax[0].imshow(rgb[id_])
        ax[0].set_title("Initial RGB Image")
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        
        ax[1].imshow(noise_type_1[id_])
        ax[1].set_title("Type-I Noise")
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        
        ax[2].imshow(noise_type_2[id_])
        ax[2].set_title("Type-II Noise")
        ax[2].set_yticklabels([])
        ax[2].set_xticklabels([])
        ax[2].set_yticks([])
        ax[2].set_xticks([])
        
        ax[3].imshow(noise_type_1_2[id_])
        ax[3].set_title("Type-I + Type-II Noise")
        ax[3].set_yticklabels([])
        ax[3].set_xticklabels([])
        ax[3].set_yticks([])
        ax[3].set_xticks([])
        
        ax[4].imshow(noise_type_2_1[id_])
        ax[4].set_title("Type-II + Type-I Noise")
        ax[4].set_yticklabels([])
        ax[4].set_xticklabels([])
        ax[4].set_yticks([])
        ax[4].set_xticks([])
        
        fig.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(1, 5, sharey=True)

        ax[0].set_title("Initial RGB Image")
        sns.kdeplot(rgb[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[0], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(rgb[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[0], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(rgb[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[0], color='blue', label='Blue Channel').set(xlim=(0, 1))

        ax[0].legend()

        
        ax[1].set_title(f"Type-I Noise")
        sns.kdeplot(noise_type_1[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[1], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_1[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[1], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_1[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[1], color='blue', label='Blue Channel').set(xlim=(0, 1))
            
        ax[1].legend()


        ax[2].set_title(f"Type-II Noise")
        sns.kdeplot(noise_type_2[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[2], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_2[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[2], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_2[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[2], color='blue', label='Blue Channel').set(xlim=(0, 1))

        ax[2].legend()
            

        ax[3].set_title(f"Type-I + Type-II Noise")
        sns.kdeplot(noise_type_1_2[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[3], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_1_2[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[3], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_1_2[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[3], color='blue', label='Blue Channel').set(xlim=(0, 1))

        ax[3].legend()
        
        ax[4].set_title(f"Type-II + Type-I Noise")
        sns.kdeplot(noise_type_2_1[id_][:, :, 0].numpy().reshape((-1)), 
                ax=ax[4], color='red', label='Red Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_2_1[id_][:, :, 1].numpy().reshape((-1)), 
                ax=ax[4], color='green', label='Green Channel').set(xlim=(0, 1))

        sns.kdeplot(noise_type_2_1[id_][:, :, 2].numpy().reshape((-1)), 
                ax=ax[4], color='blue', label='Blue Channel').set(xlim=(0, 1))

        ax[4].legend()

        plt.suptitle("Color Distribution")
        fig.text(0.5, 0.04, 'Normalized Color Value', ha='center')

        plt.show()
        
        exit()
