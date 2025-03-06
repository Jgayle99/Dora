import torch

# Load the tensor saved in output_features.pt
features = torch.load("p_256_sf-4240_view_027_depth0001_features.pt")

# Print basic information about the tensor
print("Type of features:", type(features))
print("Shape of features:", features.shape)
print("Data type:", features.dtype)

# Optionally, print a small portion of the tensor for inspection
print("First few values from the tensor slice:")
print(features[0, :5, :5])
