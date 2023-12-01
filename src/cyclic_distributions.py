import torch
import matplotlib.pyplot as plt

def create_cyclic_group_distribution(group_number, bin_size=360):
    distribution = torch.zeros(bin_size)

    if group_number == 0:
        distribution[0] = 1.
        return distribution
    elif group_number % 2 == 0:  # Even group numbers
        key_angles = [i * 360 / group_number for i in range(group_number)]
    else:  # Odd group numbers
        key_angles = [0] + [(360 / group_number) * i for i in range(1, group_number)]

    # Assign probabilities
    for angle in key_angles:
        bin_index = int((angle + 180) % 360)  # Adjust index to [-180, 180]
        # Split probability for 180 and -180 degrees
        if angle == 180 or angle == -180:
            distribution[0] += 0.5 / group_number
            distribution[-1] += 0.5 / group_number
        else:
            distribution[bin_index] += 1.0 / group_number

    return distribution

def plot_cyclic_group_distribution(group_number, bin_size=360):
    distribution = create_cyclic_group_distribution(group_number, bin_size)
    angles = torch.linspace(-180, 180, bin_size)

    plt.figure(figsize=(10, 4))
    plt.stem(angles.numpy(), distribution.numpy(), use_line_collection=True)
    plt.title(f'Cyclic Group C{group_number} Distribution')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()



def cyclic_classifier(target_rots, bin_size=360, epsilon=1e-10, n_cyclic_groups=8):
    batch_size, n_rotations = target_rots.shape
    cyclic_classification = torch.zeros(batch_size, dtype=torch.int64)

    # Create distributions for C1 to Cn
    cyclic_distributions = [create_cyclic_group_distribution(i, bin_size).cuda() for i in range(1, n_cyclic_groups + 1)]

    # Add epsilon to avoid log(0)
    for dist in cyclic_distributions:
        dist += epsilon

    for i in range(batch_size):
        # Find the prob distribution for the rotations
        hist = torch.histc(target_rots[i], bins=bin_size, min=-180, max=180)
        hist = hist / hist.sum()  # Normalize
        hist += epsilon

        # Calculate KL divergence for each cyclic group and classify
        kl_divergences = [torch.nn.functional.kl_div(torch.log(cyclic_dist), hist, reduction='sum')
                          for cyclic_dist in cyclic_distributions]
        cyclic_classification[i] = torch.argmin(torch.tensor(kl_divergences))  # classify in 0 to n_cyclic_groups
    return cyclic_classification
