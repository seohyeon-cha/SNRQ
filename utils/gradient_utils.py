import torch
from tqdm import tqdm
import os
import logging
import utils.modelutils as modelutils
import torch.nn.functional as F 

# assumes DEV is defined somewhere (from quant import * in your script)
# and that `find_layers` has the same definition as at the bottom of opt_aq.py


def get_kd_gradients(
        teacher_model,
        student_model,
        dataloader,
        num_batches=None,      # how many batches from dataloader to use
        T=1.0,                 # KD temperature
        save_path=None,
        dev='cpu'
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
        
    teacher_model.cpu()
    student_model.cpu()

    # Harvest gradients per decoder layer / module
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
                    grad = module.weight.grad.detach().clone()     # this is already squared by hook
                    grads_per_layer[module_name] = grad
                    sensitivity_per_layer[module_name] = grad.abs()
        gradients.append(grads_per_layer)
        sensitivities.append(sensitivity_per_layer)

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