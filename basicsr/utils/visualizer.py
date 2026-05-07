import torch
import numpy as np
from torchvision import transforms
from basicsr.archs import RCAN
from basicsr.utils import flow_utils

# create RCAN model
model = RCAN(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, reduction=16, scale=4)

# load pre-trained weights
model_path = 'path/to/rcan.pth'
model.load_state_dict(torch.load(model_path), strict=True)

# set the model to evaluation mode
model.eval()

# define image transform
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# load and preprocess input image
input_path = 'path/to/input_image.png'
input_img = Image.open(input_path).convert('RGB')
input_tensor = img_transform(input_img).unsqueeze(0)

# get the output feature maps of the last conv layer
with torch.no_grad():
    output_tensor = model(input_tensor)
    last_conv_layer = output_tensor[-1]

# convert tensor to numpy array and resize to the size of the input image
last_conv_layer_np = last_conv_layer.squeeze(0).cpu().numpy()
last_conv_layer_np_resized = flow_utils.tensor2img(last_conv_layer_np, out_type=np.float32, min_max=(-1, 1),
                                                  scale_255=False, normalize=False)
last_conv_layer_np_resized = np.transpose(last_conv_layer_np_resized, (1, 2, 0))
last_conv_layer_np_resized = cv2.resize(last_conv_layer_np_resized, input_img.size, interpolation=cv2.INTER_CUBIC)

# visualize the last conv layer as an RGB image
last_conv_layer_rgb = flow_utils.visualize_flow(last_conv_layer_np_resized, normalize=False,
                                                to_numpy=False, cmap=None, return_pil_image=True)

# save the visualization
vis_path = 'path/to/last_conv_layer_visualization.png'
last_conv_layer_rgb.save(vis_path)
