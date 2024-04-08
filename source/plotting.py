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