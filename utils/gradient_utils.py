import torch
from tqdm import tqdm
import os
import logging
import utils.modelutils as modelutils
import torch.nn.functional as F 

# assumes DEV is defined somewhere (from quant import * in your script)
# and that `find_layers` has the same definition as at the bottom of opt_aq.py


def get_saliency_gradients(
        model,
        dataloader,
        num_batches=None,
        num_groups=None,
        saliency_path=None,
        grd_path=None,
        dev='cpu',
        args=None
):
    """
    Calculates weight gradients for the given input tokens. Optionally also calculates
    'saliency' (mean squared gradient w.r.t. each module's output activations, grouped
    by channel) if 'saliency_path' is provided. In that case, we save one file per layer
    under 'saliency_path' directory (e.g., l0.pt, l1.pt, ...).

    This follows the GuidedQuant approach: compute gradients from end loss via backpropagation.

    Args:
        model:          Model to compute gradients for
        dataloader:     DataLoader returning batches (tokens, _) or just tokens
        num_batches:    Number of batches to process (None = all)
        num_groups:     Number of groups to chunk the channel dimension for saliency.
                       E.g. if hidden_dim=4096 and num_groups=4, each group has 1024 channels.
        saliency_path:  Directory in which to save the saliency files (one file per layer).
                       If None, no saliency is computed/saved.
        grd_path:       Path to save the final weight gradients (list of dicts).
                       If the file already exists, it will be loaded instead of recomputed.
        dev:            Device to run on
        args:           Optional args object

    Returns:
        gradients (list of dict): The list of per-layer, per-module weight gradients.
    """


    # ----------------------------------------------------------------
    # 1) Possibly load from cache (gradients only)
    # ----------------------------------------------------------------
    if grd_path is not None and os.path.isfile(grd_path):
        logging.info(f"Gradients already calculated and saved at {grd_path}.")
        logging.info(f"Loading cached gradients...")
        return torch.load(grd_path, map_location='cpu')

    logging.info(f"Calculating gradients on {len(dataloader) if hasattr(dataloader, '__len__') else 'unknown'} batches...")

    # ----------------------------------------------------------------
    # 2) Prepare model
    # ----------------------------------------------------------------
    model = model.to(dev)
    model_dtype = model.dtype
    model = model.bfloat16()
    model.train()  # Need train mode to compute gradients
    model.zero_grad(set_to_none=True)  # Zero gradients before starting

    layers = model.model.layers

    sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]

    # ----------------------------------------------------------------
    # 3) If we want saliency, set up forward hooks
    # ----------------------------------------------------------------
    # We'll store a list-of-dicts parallel to `layers`:
    #   saliency_data[i_layer][module_name] = list of [bsz, seq_len, num_groups]
    saliency_data = None
    saliency_hooks = []

    if saliency_path is not None:
        if num_groups is None:
            raise ValueError("num_groups must be provided when saliency_path is specified")
        
        # Initialize saliency_data structure
        saliency_data = []
        for layer in layers:
            full = modelutils.find_layers(layer, layers=[torch.nn.Linear])
            layer_dict = {}
            for names in sequential:
                subset = {n: full[n] for n in names if n in full}
                for module_name in subset.keys():
                    layer_dict[module_name] = []
            saliency_data.append(layer_dict)

        def make_forward_hook(layer_idx, module_name):
            def forward_hook(module, inp, out):
                # We'll store gradient on 'out', so we must retain it
                out.retain_grad()

                def grad_hook(grad):
                    """
                    grad shape typically [bsz, seq_len, hidden_dim].
                    We group the channels, compute squared gradient, then average per group.
                    """
                    bsz, seq_len, hidden_dim = grad.shape
                    group_size = hidden_dim // num_groups

                    # Compute squared gradient, scale by 1e3 as in original GuidedQuant
                    grad_squared = (grad.float() * 1e3).pow(2).view(bsz, seq_len, num_groups, group_size)
                    mean_squared_grad = grad_squared.mean(dim=-1)  # -> [bsz, seq_len, num_groups]

                    # Move to CPU and store
                    saliency_data[layer_idx][module_name].append(mean_squared_grad.bfloat16().cpu())

                # Attach the gradient hook to 'out'
                out.register_hook(grad_hook)
            return forward_hook

        # Attach hooks for all layers
        for layer_idx, layer in enumerate(layers):
            full = modelutils.find_layers(layer, layers=[torch.nn.Linear])
            for names in sequential:
                subset = {n: full[n] for n in names if n in full}
                for module_name, module in subset.items():
                    h = module.register_forward_hook(make_forward_hook(layer_idx, module_name))
                    saliency_hooks.append(h)

    # ----------------------------------------------------------------
    # 4) Weight-gradient hook (square_grad_hook)
    # ----------------------------------------------------------------
    def square_grad_hook(grad):
        return grad.pow(2)

    weight_hooks = []
    for layer in layers:
        full = modelutils.find_layers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            for module_name, module in subset.items():
                weight_hooks.append(module.weight.register_hook(square_grad_hook))

    # ----------------------------------------------------------------
    # 5) Forward/backward pass over data
    # ----------------------------------------------------------------
    if num_batches is None:
        num_batches = len(dataloader) if hasattr(dataloader, '__len__') else None

    batch_iter = iter(dataloader)
    processed = 0
    for batch in tqdm(batch_iter, desc="Calculating gradients", total=num_batches):
        if num_batches is not None and processed >= num_batches:
            break
        
        # Handle dataloader format: (tokens, _) or just tokens
        if isinstance(batch, (tuple, list)):
            tokens = batch[0]
        else:
            tokens = batch
        
        tokens = tokens.to(dev)
        # Ensure tokens have batch dimension
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        
        # Forward pass with labels for cross-entropy loss
        outputs = model(input_ids=tokens, labels=tokens)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        processed += 1

    # ----------------------------------------------------------------
    # 6) Remove hooks
    # ----------------------------------------------------------------
    for h in weight_hooks:
        h.remove()

    for h in saliency_hooks:
        h.remove()

    # ----------------------------------------------------------------
    # 7) Harvest the weight gradients (before moving to CPU)
    # ----------------------------------------------------------------
    gradients = []
    for layer in layers:
        gradients_per_layer = {}
        full = modelutils.find_layers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            for module_name, module in subset.items():
                if module.weight.grad is not None:
                    gradients_per_layer[module_name] = module.weight.grad.detach().clone().cpu()
                else:
                    gradients_per_layer[module_name] = None
        gradients.append(gradients_per_layer)

    # ----------------------------------------------------------------
    # 8) Move model back to CPU
    # ----------------------------------------------------------------
    model.cpu()
    model.eval()
    model = model.to(model_dtype)
    
    # ----------------------------------------------------------------
    # 9) Save saliency per layer, if computed
    # ----------------------------------------------------------------
    if saliency_path is not None:
        logging.info(f"Saving saliency files to {saliency_path}...")

        # Ensure directory exists
        os.makedirs(saliency_path, exist_ok=True)

        # For each layer, gather module data -> single dictionary, then save
        for layer_idx, layer in enumerate(layers):
            # Build dict of { module_name -> cat_tensor or None }
            layer_dict = {}
            for module_name, chunk_list in saliency_data[layer_idx].items():
                if len(chunk_list) > 0:
                    cat_tensor = torch.cat(chunk_list, dim=0)  # shape: [N, seq_len, num_groups]
                else:
                    cat_tensor = None
                layer_dict[module_name] = cat_tensor

            # Save each layer's dictionary to l{layer_idx}.pt
            filename = os.path.join(saliency_path, f"l{layer_idx}.pt")

            if os.path.exists(filename):
                logging.warning(f"[WARNING] File {filename} already exists. Overwriting.")

            torch.save(layer_dict, filename)

    # ----------------------------------------------------------------
    # 10) Save the gradients (if needed)
    # ----------------------------------------------------------------
    if grd_path is not None:
        logging.info(f"Saving gradients to {grd_path}...")
        if not grd_path.endswith('.pt'):
            grd_path = grd_path + '.pt'
        os.makedirs(os.path.dirname(grd_path), exist_ok=True)
        if os.path.exists(grd_path):
            logging.warning(f"[WARNING] File {grd_path} already exists. Overwriting.")
        torch.save(gradients, grd_path)

    # ----------------------------------------------------------------
    # 11) Return the gradients
    # ----------------------------------------------------------------
    return gradients


def get_kd_gradients(
        teacher_model,
        student_model,
        dataloader,
        num_batches=None,      # how many batches from dataloader to use
        T=1.0,                 # KD temperature
        save_path=None,
        dev='cpu',
        args=None
):
    """
    Compute squared-gradient accumulations of KD loss for the *quantized* student.

    Loss: KL( softmax(y_teacher / T) || softmax(y_student / T) ) * T^2

    Returns:
        gradients, sensitivities: same structure as get_opt_gradients
        gradients[i][module_name] is the (squared) grad for that layer/module.
    """
    # if save_path is not None and os.path.isfile(save_path):
    #     logging.info(f"KD gradients already calculated and saved at {save_path}.")
    #     logging.info("Loading KD gradients...")
    #     return torch.load(save_path, weights_only=False)

    logging.info("Calculating KD gradients for quantized OPT model...")

    teacher_model.eval()
    student_model.train()
    student_model.zero_grad(set_to_none=True)

    layers = student_model.model.layers

    # Hook: square the gradient before it is stored in .grad
    def grad_hook(grad):
        return grad

    hooks = []

    sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
    
    # Register hooks on the *same* modules you quantize (Linear/Conv in each decoder layer)
    for layer in layers:
        full = modelutils.find_layers(layer, layers=[torch.nn.Linear])
        for names in sequential: 
            subset = {n: full[n] for n in names}
            for module in subset.values():
                hooks.append(module.weight.register_hook(grad_hook))


    # Iterate over calibration batches and accumulate squared grads
    if num_batches is None:
        num_batches = len(dataloader)

    logging.info(f"Using {num_batches} batches from dataloader for gradient estimation.")

    teacher_model = teacher_model.to(dev)
    student_model = student_model.to(dev)

    batch_iter = iter(dataloader)
    for b_idx in tqdm(range(num_batches), desc="Calculating KD gradients"):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break

        # your dataloader from get_loaders returns (tokens, _)
        tokens = batch[0].to(dev)
        # make sure sequence length matches model.seqlen
        tokens = tokens[:, :student_model.seqlen]

        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = teacher_model(input_ids=tokens)
            # Some HF models return logits as .logits, some as last element
            teacher_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]

        # Student forward (quantized model, with grad)
        s_out = student_model(input_ids=tokens)
        student_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]

        # KL(teacher || student) with temperature T
        teacher_log_prob = F.log_softmax(teacher_logits / T, dim=-1)
        student_log_prob = F.log_softmax(student_logits / T, dim=-1)

        # classic KD scaling by T^2
        if args.reverse_kd:
            loss_kd = F.kl_div(teacher_log_prob, student_log_prob, log_target=True, reduction="batchmean") * (T * T)
        else:
            loss_kd = F.kl_div(student_log_prob, teacher_log_prob, log_target=True, reduction="batchmean") * (T * T)
        loss_kd.backward()

        # Optional: sanitize grads in-place after this batch
        g_clip = 10000.0
        for layer in layers:
            full = modelutils.find_layers(layer, layers=[torch.nn.Linear])
            for names in sequential: 
                subset = {n: full[n] for n in names}
                for module in subset.values():
                    if module.weight.grad is not None:
                        g = module.weight.grad
                        g.data = torch.nan_to_num(g.data, nan=0.0, posinf=0.0, neginf=0.0)
                        g.data.clamp_(-g_clip, g_clip)

    for hook in hooks:
        hook.remove()
    
    # Harvest gradients per decoder layer / module BEFORE moving models to CPU
    # This ensures gradients are on the correct device
    gradients = []
    sensitivities = []

    for layer in layers:
        full = modelutils.find_layers(layer, layers=[torch.nn.Linear])
        grads_per_layer = {}
        sensitivity_per_layer = {}
        for names in sequential: 
            subset = {n: full[n] for n in names}
            for module_name, module in subset.items():
                grads_per_layer[module_name] = None
                sensitivity_per_layer[module_name] = None
                if module.weight.grad is not None:
                    grad = module.weight.grad.detach().clone().cpu()
                    grads_per_layer[module_name] = grad
                    sensitivity_per_layer[module_name] = grad.abs()
        gradients.append(grads_per_layer)
        sensitivities.append(sensitivity_per_layer)
    
    # Move models to CPU after harvesting gradients
    teacher_model.cpu()
    student_model.cpu()

    stored = [gradients, sensitivities]
    if save_path is not None:
        logging.info(f"Saving KD gradients to {save_path}...")
        if not save_path.endswith('.pt'):
            save_path = save_path + '.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            logging.warning(f"[WARNING] File {save_path} already exists. Overwriting.")
        torch.save(stored, save_path)

    return stored


def summarize_layer_sensitivities(sensitivities):
    """
    Reduce per-module sensitivity tensors to a scalar score for each OPT layer.

    Args:
        sensitivities: list[layer][module_name] -> tensor

    Returns:
        list of floats where entry i is the average absolute gradient magnitude
        for layer i across its modules.
    """
    if not sensitivities:
        return []

    layer_scores = []
    for layer_sens in sensitivities:
        module_scores = []
        for tensor in layer_sens.values():
            if tensor is None:
                continue
            module_scores.append(float(tensor.float().mean().item()))
        layer_scores.append(
            sum(module_scores) / len(module_scores) if module_scores else 0.0
        )
    
    print("Layer scores length: ", len(layer_scores))
    return layer_scores


def plot_layer_sensitivities(
    sensitivities,
    save_path=None,
    show=False,
    title="OPT Layer Sensitivity",
    alphas=None,
):
    """
    Plot average sensitivity per layer as a 2D curve (layer index vs sensitivity).
    Optionally plot alpha values on the same figure with a secondary y-axis.

    Args:
        sensitivities: list returned by `get_opt_gradients` (second element)
        save_path: optional path to save the figure (PNG, PDF, etc.)
        show: if True, display the figure via plt.show()
        title: figure title
        alphas: optional list of alpha values per layer to plot on secondary y-axis

    Returns:
        The list of layer-mean sensitivities that were plotted.
    """
    layer_scores = summarize_layer_sensitivities(sensitivities)
    if not layer_scores:
        logging.warning("No sensitivity values provided; skipping plot.")
        return layer_scores

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logging.error("matplotlib is required for plotting sensitivities: %s", exc)
        return layer_scores

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot sensitivity on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Mean |grad|", color=color1)
    line1 = ax1.plot(range(len(layer_scores)), layer_scores, marker='o', color=color1, label='Sensitivity')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Plot alpha on secondary y-axis if provided
    if alphas is not None and len(alphas) == len(layer_scores):
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel("Alpha", color=color2)
        line2 = ax2.plot(range(len(alphas)), alphas, marker='s', color=color2, linestyle='--', label='Alpha')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
    else:
        ax1.legend(['Sensitivity'], loc='upper left')
    
    ax1.set_title(title)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logging.info("Saved layer sensitivity plot to %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return layer_scores

def schedule_alpha_from_sensitivity(sensitivities, alpha_min, alpha_max):
    """
    Map layer sensitivities to alpha in [alpha_min, alpha_max], assigning
    larger alphas to less sensitive layers and smaller alphas to highly
    sensitive layers.
    """
    layer_scores = summarize_layer_sensitivities(sensitivities)
    print(f'Layer scores: {layer_scores}')
    if not layer_scores:
        return None

    max_score = max(layer_scores)
    min_score = min(layer_scores)
    span = max_score - min_score
    alpha_span = alpha_max - alpha_min

    if span < 1e-12 or alpha_span <= 0:
        return [alpha_max for _ in layer_scores]

    layer_alphas = []
    for score in layer_scores:
        # invert sensitivity: small score -> large alpha, large score -> small alpha
        inverted = (max_score - score) / span
        alpha = alpha_min + inverted * alpha_span
        layer_alphas.append(alpha)
    
    return layer_alphas