import cv2
import os
import torch
import numpy as np
import torchvision

# def attention_rollout(attentions, discard_ratio=0.9, head_fusion='max', self_loop=True):
#     result = torch.eye(attentions[0].size(-1), device=attentions[0].device, dtype=attentions[0].dtype)
#     with torch.no_grad():
#         for attention in attentions:
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=1)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=1)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=1)[0]
#             else:
#                 try:
#                     idx = int(head_fusion)
#                     attention_heads_fused = attention[:, idx]
#                 except:
#                     raise f"Attention head fusion type {head_fusion} not supported"

#             # Drop the lowest attentions
#             # flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             # _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             # flat.scatter_(1, indices, 0.0)  # set the lowest attentions to 0
#             # indices = indices[indices != 0]
#             # flat[0, indices] = 0

#             I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device, dtype=attention_heads_fused.dtype)
#             if not self_loop:
#                 I = torch.zeros_like(I)
#             a = (attention_heads_fused + 1.0*I)/2
#             a = a / a.sum(dim=-1, keepdim=True)
#             result = a
#             break
    
#             result = torch.matmul(a, result)

#     result = result / result.max(dim=-1, keepdim=True)[0]
#     width = int(result.size(-1)**0.5)
#     result = result.reshape(result.size(0), result.size(1), width, width)
#     return result.float().cpu().numpy()



def rollout(attentions, discard_ratio, head_fusion, self_loop=True):
    attentions = attentions.float()
    result = torch.eye(attentions[0].size(-1), dtype=attentions[0].dtype, device=attentions[0].device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                try:
                    idx = int(head_fusion)
                    attention_heads_fused = attention[:, idx]
                except:
                    raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1), dtype=attention_heads_fused.dtype, device=attention_heads_fused.device)
            if not self_loop:
                I = torch.zeros_like(I)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[:, :, :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask / mask.max(-1)[0]
    mask = mask.reshape(mask.size(0), -1, width, width).cpu().numpy()
    return mask    

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

import matplotlib.pyplot as plt

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig("images.png", dpi=300, bbox_inches='tight')



def write_to(name, img, buffer=None):
    if buffer is None:
        path = f"visualizations/{name}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path + '.png', img)
    else:
        buffer[name].append(img)

def visualize_attention(vis_buffer, pixel_values, attentions, discard_ratio, head_fusion, n_static_tokens, self_loop=True):
    vis_masks = rollout(attentions, discard_ratio=discard_ratio, head_fusion=head_fusion, self_loop=self_loop)
    for ind, (img, vis_mask) in enumerate(zip(pixel_values[:, :3], vis_masks)):
        pil_img = torchvision.transforms.functional.to_pil_image(img.float() * img.new_tensor([0.228515625, 0.2236328125, 0.224609375]).view(3, 1, 1) + img.new_tensor([0.484375, 0.455078125, 0.40625]).view(3, 1, 1))
        np_img = np.array(pil_img)[:, :, ::-1]
        for n_token in range(vis_mask.shape[0]):
            token_mask = cv2.resize(vis_mask[n_token], (np_img.shape[1], np_img.shape[0]))
            token_mask_img = show_mask_on_image(np_img, token_mask)
            write_to(f"{ind}/tokens/token_{n_token}", token_mask_img, vis_buffer)
        if isinstance(n_static_tokens, float):
            static_mask = cv2.resize(vis_mask[:n_static_tokens].mean(0), (np_img.shape[1], np_img.shape[0]))
            dynamic_mask = cv2.resize(vis_mask[n_static_tokens:].mean(0), (np_img.shape[1], np_img.shape[0]))
            static_mask = show_mask_on_image(np_img, static_mask)
            dynamic_mask = show_mask_on_image(np_img, dynamic_mask)

            img_name = "none"
            write_to(f"{ind}/input", np_img, vis_buffer)
            write_to(f"{ind}/{img_name}_static", static_mask, vis_buffer)
            write_to(f"{ind}/{img_name}_dynamic", dynamic_mask, vis_buffer)
        elif isinstance(n_static_tokens, list):
            img_name = "none"
            write_to(f"{ind}/input", np_img, vis_buffer)
            
            static_cum = 0
            for i, _n_static_tokens in enumerate(n_static_tokens):
                static_mask = cv2.resize(vis_mask[static_cum:static_cum+_n_static_tokens].mean(0), (np_img.shape[1], np_img.shape[0]))
                static_cum += _n_static_tokens
                static_mask = show_mask_on_image(np_img, static_mask)
                write_to(f"{ind}/{img_name}_static{i}", static_mask, vis_buffer)
            dynamic_mask = cv2.resize(vis_mask[static_cum:].mean(0), (np_img.shape[1], np_img.shape[0]))
            dynamic_mask = show_mask_on_image(np_img, dynamic_mask)
            write_to(f"{ind}/{img_name}_dynamic", dynamic_mask, vis_buffer)