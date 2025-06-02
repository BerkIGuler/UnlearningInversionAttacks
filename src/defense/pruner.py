import torch
import torch.nn.utils.prune as prune
from pathlib import Path


class ModelPruner:
    """Simple unstructured pruning class for PyTorch models."""

    def __init__(self, model_class, model_kwargs=None):
        """
        Initialize the pruner.

        Args:
            model_class: The model class to instantiate (e.g., ConvNet)
            model_kwargs: Dict of kwargs for model initialization (e.g., {'width': 64})
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}

    def prune_model(self, model, pruning_amount=0.5):
        """
        Apply global unstructured pruning to all Conv2d and Linear layers.

        Args:
            model: PyTorch model to prune
            pruning_amount: Fraction of weights to prune (0.0 to 1.0)

        Returns:
            model: Pruned model (modified in-place)
        """
        parameters_to_prune = []

        # Collect all Conv2d and Linear layers
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))

        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount,
        )

        return model

    def get_sparsity(self, model):
        """
        Calculate model sparsity.

        Args:
            model: PyTorch model

        Returns:
            tuple: (sparsity_ratio, total_params, zero_params)
        """
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        zeros = sum((p == 0).sum().item() for p in model.parameters() if p.requires_grad)
        sparsity = zeros / total if total > 0 else 0
        return sparsity, total, zeros

    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            tuple: (model, checkpoint_dict)
        """
        model = self.model_class(**self.model_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        return model, checkpoint

    def save_checkpoint(self, model, checkpoint_dict, output_path, pruning_amount):
        """
        Save pruned model checkpoint.

        Args:
            model: Pruned model
            checkpoint_dict: Original checkpoint dictionary
            output_path: Path to save pruned checkpoint
            pruning_amount: Pruning amount used
        """
        sparsity, total_params, zero_params = self.get_sparsity(model)

        # Create new checkpoint
        pruned_checkpoint = {
            'model_state_dict': model.state_dict(),
            'pruning_amount': pruning_amount,
            'sparsity': sparsity,
            'total_params': total_params,
            'zero_params': zero_params
        }

        # Preserve original metadata
        if isinstance(checkpoint_dict, dict):
            for key in checkpoint_dict:
                if key not in ['model_state_dict', 'state_dict']:
                    pruned_checkpoint[key] = checkpoint_dict[key]

        torch.save(pruned_checkpoint, output_path)
        return sparsity, total_params, zero_params

    def prune_checkpoint(self, checkpoint_path, output_path, pruning_amount=0.5, verbose=True):
        """
        Prune a single checkpoint file.

        Args:
            checkpoint_path: Path to input checkpoint
            output_path: Path to save pruned checkpoint
            pruning_amount: Fraction of weights to prune
            verbose: Whether to print progress

        Returns:
            dict: Statistics about the pruning
        """
        checkpoint_path = Path(checkpoint_path)
        output_path = Path(output_path)

        if verbose:
            print(f"Pruning {checkpoint_path.name} with {pruning_amount:.0%} sparsity...")

        # Load model
        model, checkpoint_dict = self.load_checkpoint(checkpoint_path)

        # Apply pruning
        self.prune_model(model, pruning_amount)

        # Save pruned model
        sparsity, total_params, zero_params = self.save_checkpoint(
            model, checkpoint_dict, output_path, pruning_amount
        )

        if verbose:
            print(f"  Sparsity: {sparsity:.1%} ({zero_params:,}/{total_params:,} zeros)")
            print(f"  Saved: {output_path}")

        return {
            'sparsity': sparsity,
            'total_params': total_params,
            'zero_params': zero_params,
            'pruning_amount': pruning_amount
        }

    def prune_folder(self, input_folder, output_folder, pruning_amount=0.5,
                     output_prefix="pruned", verbose=True):
        """
        Prune all .pth files in a folder.

        Args:
            input_folder: Folder containing .pth files
            output_folder: Folder to save pruned files
            pruning_amount: Fraction of weights to prune
            output_prefix: Prefix for output filenames
            verbose: Whether to print progress

        Returns:
            list: List of pruning statistics for each file
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        pth_files = list(input_folder.glob('*.pth'))

        if not pth_files:
            if verbose:
                print(f"No .pth files found in {input_folder}")
            return []

        if verbose:
            print(f"Found {len(pth_files)} files. Pruning {pruning_amount:.0%} of weights...")
            print("-" * 50)

        results = []
        for pth_file in pth_files:
            output_file = output_folder / f"{output_prefix}_{pruning_amount}_{pth_file.name}"
            try:
                stats = self.prune_checkpoint(pth_file, output_file, pruning_amount, verbose)
                stats['input_file'] = str(pth_file)
                stats['output_file'] = str(output_file)
                results.append(stats)
            except Exception as e:
                if verbose:
                    print(f"Error with {pth_file.name}: {e}")
                results.append({
                    'input_file': str(pth_file),
                    'error': str(e)
                })

        if verbose:
            print("-" * 50)
            print("Done!")

        return results
