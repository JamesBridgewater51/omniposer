import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Tuple, Union
from prettytable import PrettyTable


def print_layer_info(layer, layer_idx, verbose=True):
    """Prints detailed information about a given layer.

    Args:
        layer (nn.Module): The layer to analyze.
        layer_idx (int): The index of the layer in the network.
        verbose (bool):  If True, print a lot more details (e.g., parameter values, etc.).
    """

    print(f"--- Layer {layer_idx}: {layer.__class__.__name__} ---")
    print(f"Type: {type(layer)}")

    # Parameter count
    num_params = sum(p.numel() for p in layer.parameters())
    print(f"Number of parameters: {num_params}")

    # Check if layer has trainable parameters
    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")


    # Attributes specific to different layer types
    if isinstance(layer, nn.Linear):
        print(f"  - in_features: {layer.in_features}")
        print(f"  - out_features: {layer.out_features}")
        print(f"  - bias: {layer.bias is not None}")
    elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
        print(f"  - in_channels: {layer.in_channels}")
        print(f"  - out_channels: {layer.out_channels}")
        print(f"  - kernel_size: {layer.kernel_size}")
        print(f"  - stride: {layer.stride}")
        print(f"  - padding: {layer.padding}")
        print(f"  - dilation: {layer.dilation}")
        print(f"  - groups: {layer.groups}")
        print(f"  - bias: {layer.bias is not None}")
    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        print(f"  - num_features: {layer.num_features}")
        print(f"  - eps: {layer.eps}")
        print(f"  - momentum: {layer.momentum}")
        print(f"  - affine: {layer.affine}")
        print(f"  - track_running_stats: {layer.track_running_stats}")
    elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d)):
        print(f"  - kernel_size: {layer.kernel_size}")
        print(f"  - stride: {layer.stride}")
        print(f"  - padding: {layer.padding}")
        print(f"  - dilation: {layer.dilation}")
        print(f"  - return_indices: {layer.return_indices}")
    elif isinstance(layer, nn.Embedding):
        print(f"  - num_embeddings: {layer.num_embeddings}")
        print(f"  - embedding_dim: {layer.embedding_dim}")
        print(f"  - padding_idx: {layer.padding_idx}")
    elif isinstance(layer, (nn.LSTM, nn.GRU)):
        print(f"  - input_size: {layer.input_size}")
        print(f"  - hidden_size: {layer.hidden_size}")
        print(f"  - num_layers: {layer.num_layers}")
        print(f"  - bias: {layer.bias}")
        print(f"  - batch_first: {layer.batch_first}")
        print(f"  - dropout: {layer.dropout}")
        print(f"  - bidirectional: {layer.bidirectional}")

    if verbose:
        print("\nParameters:")
        for name, param in layer.named_parameters():
            print(f"  - {name}: shape={param.shape}, requires_grad={param.requires_grad}")
            if param.requires_grad and param.grad is not None:
                print(f"    Gradient stats: mean={param.grad.mean().item():.4f}, std={param.grad.std().item():.4f}")
            if param.requires_grad:
                print(f"    Parameter stats: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")

        # Analyze weight distribution (histograms)
        if hasattr(layer, 'weight') and layer.weight is not None:
            weights = layer.weight.data.cpu().numpy().flatten() # Flatten weights for better visualization
            plt.figure(figsize=(8, 4))
            plt.hist(weights, bins=50, density=True, alpha=0.7, label='Weight values')
            plt.title(f"Weight Distribution - Layer {layer_idx} ({layer.__class__.__name__})")
            plt.xlabel("Weight Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.show() # Show the plot directly in the notebook


def analyze_module(model, input_size=None, verbose=True, show_graph=False, device="cpu"):
    """Analyzes a PyTorch nn.Module, printing detailed information about its structure and layers.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        input_size (tuple, optional): The size of the input tensor (e.g., (1, 3, 224, 224) for an image).
                                       If provided, it will perform a forward pass to analyze layer outputs.  Important for dynamically shaped layers.
        verbose (bool): If True, prints detailed information for each layer.
        show_graph (bool): If True, prints the entire computational graph (very verbose!).
        device (str): Device to put the model and input on for forward pass.  Defaults to 'cpu'.
    """

    print("--- Model Summary ---")
    print(f"Model class: {model.__class__.__name__}")

    # Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Total trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")


    # Layer-wise analysis
    print("\n--- Layer-wise Analysis ---")
    layer_idx = 0
    for name, layer in model.named_modules():  # Use named_modules() for full hierarchy
        if name == "":
            continue # skip the top level model

        print_layer_info(layer, layer_idx, verbose)
        layer_idx += 1

    # Analyze outputs and forward pass (if input_size is provided)
    if input_size:
        print("\n--- Forward Pass Analysis ---")
        try:
            # Move model to device
            model = model.to(device)
            dummy_input = torch.randn(input_size).to(device)

            # Perform forward pass with dummy input
            with torch.no_grad(): # Disable gradient calculation during the forward pass
                outputs = OrderedDict()
                x = dummy_input

                def hook_fn(module, input, output, name):  # Capture outputs with a hook
                    outputs[name] = output

                hooks = []
                layer_idx = 0
                for name, layer in model.named_modules():
                   if name == "":
                        continue

                   hook = layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
                   hooks.append(hook)
                   layer_idx+=1

                model(x)

                #Remove the hooks to prevent memory leaks
                for hook in hooks:
                    hook.remove()


            # Print output shapes
            for layer_name, output in outputs.items():
                print(f"Layer '{layer_name}' output shape: {output.shape}, dtype: {output.dtype}, device: {output.device}")
                print(f"    - Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
                if verbose:
                    try:
                        output_np = output.cpu().numpy().flatten()
                        plt.figure(figsize=(8,4))
                        plt.hist(output_np, bins=50, density=True, alpha=0.7, label='Output values')
                        plt.title(f"Output Distribution - Layer '{layer_name}'")
                        plt.xlabel("Output Value")
                        plt.ylabel("Density")
                        plt.legend()
                        plt.grid(True)
                        plt.show()
                    except Exception as e:
                        print(f"Plotting output distribution failed for layer {layer_name} with exception: {e}")


        except Exception as e:
            print(f"Error during forward pass analysis: {e}")
        finally:
            #Move the model back to the CPU to avoid GPU memory leakage if it was not originally there
            if device != "cpu":
                model = model.to("cpu")

    if show_graph:
        try:
            from torchviz import make_dot
            # Generate the computational graph
            dummy_input = torch.randn(input_size).to(device)
            y = model(dummy_input)
            dot = make_dot(y, params=dict(model.named_parameters()))
            dot.render("model_graph", view=False, format='png') # Saves to model_graph.png
            print("Computational graph saved to model_graph.png")

        except ImportError:
            print("torchviz is not installed.  Install it using `pip install torchviz` to visualize the computational graph.")
        except Exception as e:
            print(f"Failed to generate the computational graph. Error: {e}")

def model_summary(model: nn.Module, input_size: Tuple, max_depth: int = -1) -> None:
    
    # List to store summary data
    summary = []
    
    # 1. Define the hook function
    def register_hook(module):
        def hook(module, input, output):
            # Get class name
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            
            # Calculate parameters for this specific layer ONLY
            # (We use parameters(recurse=False) to ensure we don't count children)
            params = sum(p.numel() for p in module.parameters(recurse=False))
            
            output_shape = list(output.shape)
            
            summary.append({
                "Layer Type": class_name,
                "Output Shape": output_shape,
                "Param #": params
            })
            
        # Only register hooks for layers that are not containers (leaves)
        # or layers that have weights. 
        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and \
           not (module == model):
            return module.register_forward_hook(hook)

    # 2. Register hooks recursively
    hooks = []
    for name, layer in model.named_modules():
        # Skip the top-level container to avoid duplicate printing
        if layer == model:
            continue
        hooks.append(register_hook(layer))

    # 3. Perform a forward pass
    # This triggers the hooks to collect shape data
    try:
        x = torch.randn(input_size)
        with torch.no_grad():
            model(x)
    except Exception as e:
        print(f"Forward pass failed: {e}")
    finally:
        # Remove hooks to keep model clean
        for h in hooks:
            if h: h.remove()

    # 4. Generate Table
    table = PrettyTable()
    table.field_names = ["Layer (type)", "Output Shape", "Param #"]
    table.align["Layer (type)"] = "l"
    table.align["Output Shape"] = "c"
    table.align["Param #"] = "r"

    total_params = 0
    trainable_params = 0
    
    # Calculate global totals correctly
    global_total = sum(p.numel() for p in model.parameters())
    global_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for layer in summary:
        # Note: If you sum 'layer["Param #"]' here, it might be slightly different 
        # from global_total if shared weights exist, but it's usually accurate enough 
        # for a summary view.
        table.add_row([
            layer["Layer Type"],
            str(layer["Output Shape"]),
            f"{layer['Param #']:,}"
        ])

    print(table)
    print("=" * 30)
    print(f"Total params: {global_total:,}")
    print(f"Trainable params: {global_trainable:,}")
    print(f"Non-trainable params: {global_total - global_trainable:,}")

if __name__ == '__main__':
    # Example Usage
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 14 * 14, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            return x

    # Create an instance of the model
    model = SimpleNet()
    # Call the function with the model
    analyze_module(model)