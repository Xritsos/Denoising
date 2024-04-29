import matplotlib.pyplot as plt


def visualize_sets(train_set, val_set, test_set, classes):
    # calculate counts of each class per set
    train_class_count = {}
    for _, index in train_set:
        label = classes[index]
        
        if label not in train_class_count:
            train_class_count[label] = 0
        
        train_class_count[label] += 1
    
    val_class_count = {}
    for _, index in val_set:
        label = classes[index]
        
        if label not in val_class_count:
            val_class_count[label] = 0
        
        val_class_count[label] += 1
        
    test_class_count = {}
    for _, index in test_set:
        label = classes[index]
        
        if label not in test_class_count:
            test_class_count[label] = 0
        
        test_class_count[label] += 1
        
    # train set visualization
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
    
    # Horizontal Bar Plot
    ax.barh(list(train_class_count.keys()), list(train_class_count.values()))
    
    # Remove axes splines
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    
    # Add x, y gridlines
    ax.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.0)
    
    # Show top values 
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                str(round((i.get_width()), 2)),
                fontsize=10, fontweight='bold',
                color='grey')
    
    # Add Plot Title
    ax.set_title('Train Set Classes Size',
                loc='center')

    # Show Plot
    plt.show()
    
    # validation set visualization
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
    
    # Horizontal Bar Plot
    ax.barh(list(val_class_count.keys()), list(val_class_count.values()))
    
    # Remove axes splines
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    
    # Add x, y gridlines
    ax.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.0)
    
    # Show top values 
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                str(round((i.get_width()), 2)),
                fontsize=10, fontweight='bold',
                color='grey')
    
    # Add Plot Title
    ax.set_title('Validation Set Classes Size',
                loc='center')

    # Show Plot
    plt.show()
    
    # test set visualization
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
    
    # Horizontal Bar Plot
    ax.barh(list(test_class_count.keys()), list(test_class_count.values()))
    
    # Remove axes splines
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    
    # Add x, y gridlines
    ax.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.0)
    
    # Show top values 
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                str(round((i.get_width()), 2)),
                fontsize=10, fontweight='bold',
                color='grey')
    
    # Add Plot Title
    ax.set_title('Test Set Classes Size',
                loc='center')

    # Show Plot
    plt.show()
    
    
def plot_images_kdes():
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(inputs.cpu())
    axs[0].set_title("Initial RGB Image")
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].set_yticks([])
    axs[0].set_xticks([])

    axs[1].imshow(multi_noisy.cpu())
    axs[1].set_title(f"Multiplicative Noise mean={multi_mean}, std={multi_std}")
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[1].set_yticks([])
    axs[1].set_xticks([])

    axs[2].imshow(gauss_noisy.cpu())
    axs[2].set_title(f"Gaussian Noise mean={g_mean}, std={g_std}")
    axs[2].set_yticklabels([])
    axs[2].set_xticklabels([])
    axs[2].set_yticks([])
    axs[2].set_xticks([])

    axs[3].imshow(poisson_noisy.cpu())
    axs[3].set_title(f"Poisson Noise")
    axs[3].set_yticklabels([])
    axs[3].set_xticklabels([])
    axs[3].set_yticks([])
    axs[3].set_xticks([])

    plt.show()


    fig, ax = plt.subplots(1, 4, sharey=True)

    ax[0].set_title("Initial RGB Image")
    sns.kdeplot(inputs[:, :, 0].cpu().numpy().reshape((-1)), 
            ax=ax[0], color='red', label='Red Channel').set(xlim=(0, 1))

    sns.kdeplot(inputs[:, :, 1].cpu().numpy().reshape((-1)), 
            ax=ax[0], color='green', label='Green Channel').set(xlim=(0, 1))

    sns.kdeplot(inputs[:, :, 2].cpu().numpy().reshape((-1)), 
            ax=ax[0], color='blue', label='Blue Channel').set(xlim=(0, 1))

    ax[0].legend()


    ax[1].set_title(f"Multiplicative Noise mean={multi_mean}, std={multi_std}")
    sns.kdeplot(multi_noisy[:, :, 0].cpu().numpy().reshape((-1)), 
            ax=ax[1], color='red', label='Red Channel').set(xlim=(0, 1))

    sns.kdeplot(multi_noisy[:, :, 1].cpu().numpy().reshape((-1)), 
            ax=ax[1], color='green', label='Green Channel').set(xlim=(0, 1))

    sns.kdeplot(multi_noisy[:, :, 2].cpu().numpy().reshape((-1)), 
            ax=ax[1], color='blue', label='Blue Channel').set(xlim=(0, 1))
        
    ax[1].legend()


    ax[2].set_title(f"Gaussian Noise mean={g_mean}, std={g_std}")
    sns.kdeplot(gauss_noisy[:, :, 0].cpu().numpy().reshape((-1)), 
            ax=ax[2], color='red', label='Red Channel').set(xlim=(0, 1))

    sns.kdeplot(gauss_noisy[:, :, 1].cpu().numpy().reshape((-1)), 
            ax=ax[2], color='green', label='Green Channel').set(xlim=(0, 1))

    sns.kdeplot(gauss_noisy[:, :, 2].cpu().numpy().reshape((-1)), 
            ax=ax[2], color='blue', label='Blue Channel').set(xlim=(0, 1))

    ax[2].legend()
        

    ax[3].set_title(f"Poisson Noise")
    sns.kdeplot(poisson_noisy[:, :, 0].cpu().numpy().reshape((-1)), 
            ax=ax[3], color='red', label='Red Channel').set(xlim=(0, 1))

    sns.kdeplot(poisson_noisy[:, :, 1].cpu().numpy().reshape((-1)), 
            ax=ax[3], color='green', label='Green Channel').set(xlim=(0, 1))

    sns.kdeplot(poisson_noisy[:, :, 2].cpu().numpy().reshape((-1)), 
            ax=ax[3], color='blue', label='Blue Channel').set(xlim=(0, 1))

    ax[3].legend()

    plt.suptitle("Color Distribution")
    fig.text(0.5, 0.04, 'Normalized Color Value', ha='center')

    plt.show()