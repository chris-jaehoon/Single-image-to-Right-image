# -*- coding: utf-8 -*-
"""CS231A_Final_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yZ48ukggqn8W9bNvtAbhZHuc5BhM9SiS

# Install and Import Needed Packages
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# Enter the foldername in your Drive where you have saved the unzipped
# '.py' files from the p2 folder
# e.g. 'cs231a/pset3/p2'
FOLDERNAME = 'cs231A_final_project'

assert FOLDERNAME is not None, "[!] Enter the foldername."

# %cd drive/MyDrive
# %cd $FOLDERNAME

# Install required packages. 
!pip install -qq -U diffusers==0.11.1 transformers ftfy gradio accelerate
!pip install --upgrade paddlepaddle
!pip install --upgrade paddlehub

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2

# https://huggingface.co/spaces/PaddlePaddle/U2Net
import torch as th
import paddlehub as hub

"""# Load Models"""

matting_model = hub.Module(name='U2Net')

# display_image(fg_alpha_matte(np.array(test_image))[0]['mask'])

device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"

outpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)

text_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
text_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
text_model.to(device)

# We use Vision Transformers for Depth Prediction
# TODO: How was it trained?
# https://huggingface.co/Intel/dpt-hybrid-midas

depth_feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

"""#Helper Functions"""

def display_image(image):
  imgplot = plt.imshow(image, cmap="gray")
  plt.show()

def download_image(url):
  response = requests.get(url)
  return Image.open(BytesIO(response.content)).convert("RGB")

def pad_image(image, expected_shape=(512, 512)):
  # Generate Mask
  pad_length = 100
  width, height = image.size
  new_width = width + pad_length + pad_length
  new_height = height + pad_length + pad_length

  border_padding = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
  inner_image_mask = Image.new(image.mode, (width, height), (0, 0, 0))
  
  padded_image = border_padding.copy()
  padded_image.paste(image, (pad_length, pad_length))
  padded_image = padded_image.resize(expected_shape)

  image_mask = border_padding.copy()
  image_mask.paste(inner_image_mask, (pad_length, pad_length))
  image_mask = image_mask.resize(expected_shape)

  return padded_image, image_mask

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

"""# Load Local Images"""

disparity = np.genfromtxt('disparity.csv', delimiter=",")
outpainted_image = Image.open(r"outpainted_image.png")
test_image = Image.open(r"test_matting.png")

# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
img_url = "https://ichef.bbci.co.uk/news/640/cpsprodpb/160B4/production/_103229209_horsea.png"
input_image = download_image(img_url)
input_image = input_image.resize((512, 512))
padded_image, image_mask = pad_image(input_image)

"""# Image to Text"""

def image_to_text(image):
  kwargs = {"max_length": 20, "num_beams": 4}

  pixel_values = text_feature_extractor(images=[image], return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)
  output_ids = text_model.generate(pixel_values, **kwargs)

  labels = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  labels = [label.strip() for label in labels]
  return labels[0]

"""# Apply Outpainting"""

def outpaint_image(image, image_mask):
  # TODO: Generate text description from image
  prompt = "captured with a camera is a " + image_to_text(image)
  print(prompt)
  guidance_scale=7.5
  num_samples = 3
  generator = torch.Generator(device=device).manual_seed(20) # change the seed to get different results

  images = outpaint_model(
      prompt=prompt,
      image=image,
      mask_image=image_mask,
      guidance_scale=guidance_scale,
      generator=generator,
      num_images_per_prompt=num_samples,
  ).images

  return images

image_grid([input_image], 1, 1)

images = outpaint_image(padded_image, image_mask)
image_grid(images, 1, 3)

outpainted_image = images[0]
outpainted_image.save("outpainted_image.png","PNG")

figure_1 = [input_image, padded_image, image_mask, outpainted_image]
image_grid(figure_1, 1, len(figure_1))

"""# Monocular Depth Estimation"""

from PIL import ImageFilter

def get_diparity(outpainted_image, gaussian_blur=True):
  features = depth_feature_extractor(images=outpainted_image, return_tensors="pt")

  with torch.no_grad():
      output = depth_model(**features)
      predicted_depth = output.predicted_depth

  # import pdb
  # pdb.set_trace()

  # interpolate to original size
  prediction = torch.nn.functional.interpolate(
      predicted_depth.unsqueeze(1),
      size=outpainted_image.size[::-1],
      mode="bicubic",
      align_corners=False,
  )

  # visualize the prediction
  disparity = prediction.squeeze().cpu().numpy()

  # Apply gaussian blurr
  if gaussian_blur:
    disparity = cv2.GaussianBlur(disparity, (5,5), cv2.BORDER_DEFAULT)
    pooling = th.nn.MaxPool1d(kernel_size=5, stride=1)
    disparity = pooling(th.from_numpy(disparity).float())
    disparity = cv2.resize(np.array(disparity), (512, 512), interpolation = cv2.INTER_AREA)

  disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
  # disparity += abs(disparity.min())
  # disparity /= disparity.max()
  np.savetxt('disparity.csv', disparity, delimiter=',') 
  return disparity

# disparity = get_diparity(outpainted_image, False)
# display_image(disparity)
disparity = get_diparity(outpainted_image)
display_image(disparity)
# Image.fromarray(np.clip(disparity * 255, 0, 255).astype(np.uint8)).resize((512, 512))

"""

# Soft FG Pixel Visibility"""

from scipy import ndimage
import numpy as np
from tqdm import tqdm

def fg_pixel_visibility_map(depth):
  # sobel_gradient = ndimage.laplace(depth)
  sobel_gradient_x = ndimage.sobel(depth, axis=0, mode='constant')
  sobel_gradient_y = ndimage.sobel(depth, axis=1, mode='constant')
  sobel_gradient = np.hypot(sobel_gradient_x, sobel_gradient_y)
  beta = 10
  pixel_visibility_map = np.exp(-beta * np.square(sobel_gradient))
  return pixel_visibility_map

display_image(fg_pixel_visibility_map(disparity))

"""#Soft Disocclusions"""

pad_length = 50
rho = 0.00001
gamma = 5

import jax
import jax.numpy as jnp

def get_kernel(k):
  print(k%2)
  assert k % 2 == 1
  base = jnp.zeros((k, k))
  kernels = []

  # ltr
  for i in range(k):
    if i != k // 2:
      kernels.append(base.at[k // 2, i].set(1))
      kernels.append(base.at[i, k // 2].set(1))
  return jnp.array(kernels)

import torch
from torch import nn

conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, bias=False, padding_mode='replicate', padding='same')
kernel = torch.from_numpy(np.array(get_kernel(5)[:, None]))
print(conv.weight.data.shape,  kernel.shape)
assert conv.weight.data.shape == kernel.shape
conv.weight.data = kernel


def get_disocclusion(depth_image):
  h,w = depth_image.shape
  assert h == w
  xs,ys = jnp.meshgrid(jnp.arange(h), jnp.arange(h))
  coords = jnp.stack([ys, xs], axis=-1)
  # return coords

  i_coords = coords[None, None, ..., 0]
  i_coords_t = torch.from_numpy(np.array(i_coords)).to(torch.float32)

  
  j_coords = coords[None, None, ..., 1]
  j_coords_t = torch.from_numpy(np.array(j_coords)).to(torch.float32)


  depth_image_t = torch.from_numpy(np.array(depth_image[None,None])).to(torch.float32)
  # return dists_t

  neighborhood_depths = conv(depth_image_t).detach().numpy()

  neighborhood_is = conv(i_coords_t).detach().numpy()
  neighborhood_js = conv(j_coords_t).detach().numpy()
  
  neighborhood_dists = (coords - jnp.stack([neighborhood_is, neighborhood_js], axis=-1))
  neighborhood_dists = jnp.sqrt((neighborhood_dists**2).sum(axis=-1))

  occ = (depth_image[None,None] - neighborhood_depths - rho * neighborhood_dists)
  return np.tanh(occ.max(axis=1) * gamma)

  # return neighborhood_dists

def soft_disocclusions(disparity):
  height, width = disparity.shape
  soft_disocclusion_map = np.zeros((height, width))
  for i in range(height):
    for j in range(width):
      pixel_occlusion_values = np.array([])
      # Calculate occlusion values for pixels at the top of the reference pixel
      vertical_disocclusion_values = disparity[i, j] - disparity[i, max(0, j-pad_length):j]
      vertical_disocclusion_values -= rho * (j - np.arange(max(0, j-pad_length), j))
      pixel_occlusion_values = np.append(pixel_occlusion_values, vertical_disocclusion_values)

      # Calculate occlusion values for pixels at the bottom of the reference pixel
      vertical_disocclusion_values = disparity[i, j] - disparity[i, j+1 : min(width, j+pad_length)]
      vertical_disocclusion_values -= rho * (np.arange(j+1, min(width, j+pad_length)) - j)
      pixel_occlusion_values = np.append(pixel_occlusion_values, vertical_disocclusion_values)

      # Calculate occlusion values for pixels at the left of the reference pixel
      horizontal_disocclusion_values = disparity[i, j] - disparity[max(0, i-pad_length):i, j]
      horizontal_disocclusion_values -= rho * (i - np.arange(max(0, i-pad_length), i))
      pixel_occlusion_values = np.append(pixel_occlusion_values, horizontal_disocclusion_values)

      # Calculate occlusion values for pixels at the right of the reference pixel
      horizontal_disocclusion_values = disparity[i, j] - disparity[i+1 : min(height, i+pad_length), j]
      horizontal_disocclusion_values -= rho * (np.arange(i+1, min(height, i+pad_length)) - i)
      pixel_occlusion_values = np.append(pixel_occlusion_values, horizontal_disocclusion_values)

      soft_disocclusion_map[i, j] = max(np.tanh(gamma * max(pixel_occlusion_values)), 0)
  return soft_disocclusion_map

import cv2
new_disparity = cv2.resize(disparity, (128, 128), interpolation = cv2.INTER_AREA)
display_image(new_disparity)
display_image(soft_disocclusions(new_disparity))
# display_image(soft_occlusions(new_disparity))

pad_length = 50
rho = 0.001
gamma = 5
disocc = get_disocclusion(new_disparity)
# disocc.min(), disocc.max()

Image.fromarray(np.clip(disocc[0] * 255, 0, 255).astype(np.uint8)).resize((512, 512))

"""# Soft Occlusions Map"""

# TODO: Fix this it appears to be wrong
def soft_occlusions(disparity):
  height, width = disparity.shape
  soft_disocclusion_map = np.zeros((height, width))
  for i in range(height):
    for j in range(width):
      pixel_occlusion_values = np.array([])
      # Calculate occlusion values for pixels at the top of the reference pixel
      vertical_disocclusion_values = disparity[i, j] - disparity[i, max(0, j-pad_length):j]
      vertical_disocclusion_values = rho * (j - np.arange(max(0, j-pad_length), j)) - vertical_disocclusion_values
      pixel_occlusion_values = np.append(pixel_occlusion_values, vertical_disocclusion_values)

      # Calculate occlusion values for pixels at the bottom of the reference pixel
      vertical_disocclusion_values = disparity[i, j] - disparity[i, j+1 : min(width, j+pad_length)]
      vertical_disocclusion_values = rho * (np.arange(j+1, min(width, j+pad_length)) - j) - vertical_disocclusion_values
      pixel_occlusion_values = np.append(pixel_occlusion_values, vertical_disocclusion_values)

      # Calculate occlusion values for pixels at the left of the reference pixel
      horizontal_disocclusion_values = disparity[i, j] - disparity[max(0, i-pad_length):i, j]
      horizontal_disocclusion_values = rho * (i - np.arange(max(0, i-pad_length), i)) - horizontal_disocclusion_values
      pixel_occlusion_values = np.append(pixel_occlusion_values, horizontal_disocclusion_values)

      # Calculate occlusion values for pixels at the right of the reference pixel
      horizontal_disocclusion_values = disparity[i, j] - disparity[i+1 : min(height, i+pad_length), j]
      horizontal_disocclusion_values = rho * (np.arange(i+1, min(height, i+pad_length)) - i) - horizontal_disocclusion_values
      pixel_occlusion_values = np.append(pixel_occlusion_values, horizontal_disocclusion_values)

      soft_disocclusion_map[i, j] = max(np.tanh(gamma * min(pixel_occlusion_values)), 0)
  return soft_disocclusion_map

"""# View Outputs"""

import cv2
new_disparity = cv2.resize(disparity, (128, 128), interpolation = cv2.INTER_AREA)
display_image(new_disparity)
display_image(soft_disocclusions(new_disparity))
# display_image(soft_occlusions(new_disparity))

print("Input Image")
display_image(input_image)
print("Outpainted Image")
display_image(outpainted_image)
print("Disparity Map")
display_image(disparity)
print("Soft Disocclusion Map")
display_image(soft_disocclusions(disparity))
print("Soft Occlusion Map")
display_image(soft_occlusions(disparity))
print("FG Visibility Map")
display_image(fg_pixel_visibility_map(disparity))

"""#Improved Layering with Segmentation

TODO: Fix soft occlusion function
"""

def fg_alpha_matte(image):
  output = matting_model.Segmentation(
        images=[image],
        paths=None,
        batch_size=1,
        input_size=320,
        output_dir='output',
        visualization=True)
  return output

output = fg_alpha_matte(np.array(outpainted_image))
foreground_segmentation = output[0]['mask']
display_image(foreground_segmentation)

"""# Foreground Layer"""

matte = fg_alpha_matte(np.array(outpainted_image))
foreground_segmentation = matte[0]['mask']

M = foreground_segmentation
pooling = th.nn.MaxPool1d(kernel_size=5, stride=1)
M_prime = pooling(th.from_numpy(M).float())
M_prime = cv2.resize(np.array(M_prime), (512, 512), interpolation = cv2.INTER_AREA)

occlusion_map = soft_occlusions(disparity)

depth_based_visibility_map = fg_pixel_visibility_map(disparity)

A = depth_based_visibility_map*(1 - ((M_prime - M) * (1 - occlusion_map )))
display_image(A)

RGBDA = np.zeros((512, 512, 5))
RGBDA[:, :, :3] = outpainted_image
RGBDA[:, :, 3] = disparity
RGBDA[:, :, 4] = A

outpainted_image

"""# Background Layer"""

def generate_background(outpainted_image, M_prime):
  prompt = "background of a scene" #Nice
  guidance_scale=7.5
  num_samples = 3
  generator = torch.Generator(device=device).manual_seed(20) # change the seed to get different results

  images = outpaint_model(
      prompt=prompt,
      image=outpainted_image,
      mask_image=M_prime,
      guidance_scale=guidance_scale,
      generator=generator,
      num_images_per_prompt=num_samples,
  ).images
  return images

images = generate_background(outpainted_image, M_prime)
background_rgb = images[0]
background_depth = get_diparity(background_rgb)

RGBD_prime = np.zeros((512, 512, 4))
RGBD_prime[:, :, 3] = background_depth
RGBD_prime[:, :, :3] = background_rgb

# Display background RGB and Depth
display_image(RGBD_prime[:, :, 3])
image_grid([background_rgb], 1, 1)

RGBD_prime[:, :, :3].shape

a = np.array(outpainted_image).astype(np.float32)
b = np.array(depth_based_visibility_map)

a[:, :, 0] *= b
a[:, :, 1] *= b
a[:, :, 2] *= b

fg_layer = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((512, 512))

fg_layer

dp = Image.fromarray(np.clip(depth_based_visibility_map*255, 0, 255).astype(np.uint8)).resize((512, 512))



figure_2 = [outpainted_image, Image.fromarray(np.clip(disparity*255, 0, 255).astype(np.uint8)).resize((512, 512)), dp, fg_layer]
image_grid(figure_2, 1, len(figure_2))

figure_3 = [outpainted_image, Image.fromarray(np.clip(M_prime, 0, 255).astype(np.uint8)).resize((512, 512)), background_rgb]
image_grid(figure_3, 1, len(figure_3))