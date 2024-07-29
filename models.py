import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
# import paddlehub as hub
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import cv2
from utils import get_image_file_paths, display_image, pad_image, normalize_array
import os
import logging
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
logging.basicConfig(level=logging.INFO)


class ImageToTextModel:
    def __init__(self) -> None:
        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        self.text_model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.text_feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.text_model.to(self.device)

    def describe_image(self, image: Image.Image) -> str:
        """
        Describe image using text model.

        Args:
            image: PIL image
        Returns:
            text: String
        """
        kwargs = {"max_length": 400, "num_beams": 1}

        pixel_values = self.text_feature_extractor(
            images=[image], return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = self.text_model.generate(pixel_values, **kwargs)

        labels = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        labels = [label.strip() for label in labels]
        return labels[0]


class ImageCompletionModel:
    """
    Image completion model class.
    """

    def __init__(self) -> None:
        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        model_path = "runwayml/stable-diffusion-inpainting"
        self.image_completion_model = 
        
        
        # self.image_completion_model = StableDiffusionInpaintPipeline.from_pretrained(
        #     model_path,
        #     revision="fp16",
        #     torch_dtype=th.float16,
        # ).to(self.device)

        self.image_to_text_model = ImageToTextModel()

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        inpaint image using mask and stable diffusion inpainting model.

        Args:
            image: PIL image
            mask: Numpy array
        Returns:
            inpainted_image: PIL image
        """

        logging.info("Inpainting image.")

        input_shape = np.array(image).shape[:2][::-1]
        image = image.resize((512, 512))
        mask = Image.fromarray(
            np.clip(np.sqrt(mask) * 1000, 0, 255).astype(np.uint8)
        ).resize((512, 512))

        prompt = "inpaint from background"
        guidance_scale = 10
        num_samples = 1
        generator = th.Generator(device=self.device).manual_seed(10)

        model_output = self.image_completion_model(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images[0]

        inpainted_image = model_output.resize(input_shape)

        return inpainted_image

    def outpaint(
        self, input_image: Image.Image, padded_image: Image.Image, mask: np.ndarray
    ) -> Image.Image:
        """
        Outpaint image using mask, stable diffusion inpainting model, and text prompt.

        Args:
            image: PIL image
            mask: Numpy array
        Returns:
            image: PIL image
        """
        logging.info("Outpainting image.")

        input_shape = np.array(padded_image).shape[:2][::-1]
        padded_image = padded_image.resize((512, 512))
        mask = Image.fromarray(
            np.clip(np.sqrt(mask) * 1000, 0, 255).astype(np.uint8)
        ).resize((512, 512))

        outpaint_prompt = self.image_to_text_model.describe_image(input_image)
        print(outpaint_prompt)

        guidance_scale = 7.5
        num_samples = 1
        generator = th.Generator(device=self.device).manual_seed(1)

        model_output = self.image_completion_model(
            prompt=outpaint_prompt,
            image=padded_image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images[0]

        padded_image = model_output.resize(input_shape)

        return padded_image


class MattingModel:
    """
    Matting model class.
    """

    def __init__(self) -> None:
        self.matting_model = hub.Module(name="U2Net")

    def get_alpha_matte(self, image: Image.Image) -> np.ndarray:
        """
        Get alpha matte for image.

        Args:
            image: PIL image
        Returns:
            alpha_matte: Numpy array
        """
        logging.info("Generating alpha matte.")

        image = np.asarray(image)
        alpha_matte = self.matting_model.Segmentation(
            images=[image],
            paths=None,
            batch_size=1,
            input_size=312,
            output_dir="output",
            visualization=False,
        )[0]["mask"]

        return alpha_matte


class MonocularDepthModel:
    """
    Monocular depth model class.
    """

    def __init__(self) -> None:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(th.load(f'/nfs/home/wldn1677/lgd/Single-image-to-Right-image/ckpts/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        # model_type = "DPT_Large"
        # self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        # self.midas_model = th.hub.load("intel-isl/MiDaS", model_type)
        # self.midas_model.to(self.device)
        # self.midas_model.eval()

        midas_transforms = th.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    def get_depth_map(
        self, image: Image.Image, gaussian_blur=True, normalize=True
    ) -> np.ndarray:
        """
        Get depth map for image.

        Args:
            image: PIL image
        Returns:
            depth_map: Numpy array
        """

        logging.info("Generating monocular depth image.")

        # image = np.asarray(image)
        # input_batch = self.transform(image).to(self.device)
        # input_shape = np.array(image).shape[:2][::-1]

        # with th.no_grad():
        #     prediction = self.midas_model(input_batch)

        #     prediction = th.nn.functional.interpolate(
        #         prediction.unsqueeze(1),
        #         size=image.shape[:2],
        #         mode="bicubic",
        #         align_corners=False,
        #     ).squeeze()
        input_shape = (np.asarray(image).shape[1],np.asarray(image).shape[0])
        prediction =self.model.infer_image(image)
        disparity = prediction

        # Apply gaussian blurr and max pooling.
        if gaussian_blur:
            disparity = cv2.GaussianBlur(disparity, (5, 5), cv2.BORDER_DEFAULT)
            pooling = th.nn.MaxPool1d(kernel_size=5, stride=1)
            disparity = pooling(th.from_numpy(disparity).float())

        # generate depth while ignoring extremly distant values
        depth_map = 1 / np.maximum(disparity, 2)
        depth_map = cv2.resize(
            np.array(depth_map), input_shape, interpolation=cv2.INTER_AREA
        )

        if normalize:
            depth_map = normalize_array(depth_map)

        return depth_map


# load models.
image_completion_model = ImageCompletionModel()
matting_model = MattingModel()
monocular_depth_model = MonocularDepthModel()

if __name__ == "__main__":
    image_name = get_image_file_paths()[0]
    image_path = os.path.join(os.getcwd(), image_name)
    image = Image.open(image_path).resize((400, 400))
    display_image(image)

    # Get alpha matte.
    alpha_matte = matting_model.get_alpha_matte(image)
    display_image(alpha_matte)

    # Get inpainted image.
    inpainted_image = image_completion_model.inpaint(image, alpha_matte)
    display_image(inpainted_image)

    # Get outpainted image.
    padded_image, image_mask = pad_image(image)
    outpainted_image = image_completion_model.outpaint(image, padded_image, image_mask)
    display_image(outpainted_image)

    # Get depth map.
    depth_map = monocular_depth_model.get_depth_map(image)
    display_image(depth_map)

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=F.elu):
        super(GatedConv, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)

    def forward(self, x):
        conv = self.conv(x)
        gate = torch.sigmoid(self.gate(x))
        return self.activation(conv) * gate

class ContextualAttention(nn.Module):
    def forward(self, f, b, mask):
        """ Contextual attention layer implementation
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for the foreground.
        Returns:
            tf.Tensor: output tensor.
        """
        # For simplicity, this example uses a placeholder implementation.
        # You would replace this with the actual contextual attention logic.
        return f, None

class InpaintCAModel(nn.Module):
    def __init__(self):
        super(InpaintCAModel, self).__init__()

        self.stage1 = nn.Sequential(
            GatedConv(5, 48, 5, 1),
            GatedConv(48, 48, 3, 1),
            GatedConv(48, 96, 3, 2),
            GatedConv(96, 96, 3, 1),
            GatedConv(96, 192, 3, 1),
            GatedConv(192, 192, 3, 2),
            GatedConv(192, 192, 3, 4),
            GatedConv(192, 192, 3, 8),
            GatedConv(192, 192, 3, 16),
            GatedConv(192, 192, 3, 1),
            GatedConv(192, 192, 3, 1)
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.stage1_upsample = nn.Sequential(
            GatedConv(192, 96, 3, 1),
            GatedConv(96, 96, 3, 1)
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.stage1_final = nn.Sequential(
            GatedConv(96, 48, 3, 1),
            GatedConv(48, 24, 3, 1),
            GatedConv(24, 4, 3, 1)
        )

        self.stage2_conv = nn.Sequential(
            GatedConv(4, 48, 5, 1),
            GatedConv(48, 48, 3, 1),
            GatedConv(48, 96, 3, 2),
            GatedConv(96, 96, 3, 1),
            GatedConv(96, 192, 3, 1),
            GatedConv(192, 192, 3, 2),
            GatedConv(192, 192, 3, 4),
            GatedConv(192, 192, 3, 8),
            GatedConv(192, 192, 3, 16)
        )

        self.attention_branch = nn.Sequential(
            GatedConv(4, 48, 5, 1),
            GatedConv(48, 48, 3, 2),
            GatedConv(48, 96, 3, 1),
            GatedConv(96, 192, 3, 2),
            GatedConv(192, 192, 3, 1),
            GatedConv(192, 192, 3, 1)
        )

        self.contextual_attention = ContextualAttention()

        self.stage2_final = nn.Sequential(
            GatedConv(384, 192, 3, 1),
            GatedConv(192, 192, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            GatedConv(192, 96, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            GatedConv(96, 48, 3, 1),
            GatedConv(48, 24, 3, 1),
            GatedConv(24, 3, 3, 1)
        )

    def forward(self, x, mask):
        xin = x
        ones_x = torch.ones_like(x)[:, :1, :, :]
        x = torch.cat([x, ones_x, ones_x * mask], dim=1)

        # Stage 1
        x = self.stage1(x)
        x = self.upsample1(x)
        x = self.stage1_upsample(x)
        x = self.upsample2(x)
        x_stage1 = self.stage1_final(x)

        # Stage 2
        x = x_stage1 * mask + xin[:, :3, :, :] * (1. - mask)
        x = self.stage2_conv(x)

        x_attention = self.attention_branch(xin)
        x_attention, offset_flow = self.contextual_attention(x_attention, x_attention, mask)

        x = torch.cat([x, x_attention], dim=1)
        x = self.stage2_final(x)
        x_stage2 = torch.tanh(x)

        return x_stage1, x_stage2, offset_flow


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            dis_conv(3, 64, 3, 1),
            dis_conv(64, 128, 3, 1),
            dis_conv(128, 256, 3, 1),
            dis_conv(256, 512, 3, 1),
            dis_conv(512, 512, 3, 1),
            dis_conv(512, 512, 3, 1)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return x


# Helper function for dis_conv
def dis_conv(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
        nn.LeakyReLU(0.2, inplace=True)
    )

# Example usage
if __name__ == '__main__':
    inpaint_model = InpaintCAModel()
    discriminator = Discriminator()
    x = torch.randn(4, 3, 256, 256)
    mask = torch.randn(4, 1, 256, 256)

    x_stage1, x_stage2, offset_flow = inpaint_model(x, mask)
    disc_out = discriminator(x_stage2)

    print(f'x_stage1 shape: {x_stage1.shape}')
    print(f'x_stage2 shape: {x_stage2.shape}')
    print(f'disc_out shape: {disc_out.shape}')
