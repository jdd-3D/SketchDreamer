import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTextModel

def main():
    # 加载 CLIP 处理器和模型
    clip = "openai/clip-vit-large-patch14"#-336
    processor = CLIPProcessor.from_pretrained(clip)
    model = CLIPVisionModelWithProjection.from_pretrained(clip)
    model2 = CLIPTextModel.from_pretrained(clip)

    # inputs = processor(text=["a photo of a cat"], return_tensors="pt", padding=True)
    # # print(inputs)
    # output = model2(inputs.input_ids)
    # print(len(output))
 


    # 打开图像
    image = Image.open("guidance/499.png")  # 请将路径替换为你要编码的图像的实际路径
    # print(image)
 
    # 使用处理器对图像进行预处理
    inputs = processor(images=image, return_tensors="pt").pixel_values
    print(inputs.shape)


    # 使用模型对预处理后的图像进行编码
    with torch.no_grad():
        outputs = model(inputs)
        print(outputs)
        image_embeds = outputs.last_hidden_state


    print(image_embeds.shape)


if __name__ == "__main__":
    main()

# from PIL import Image
# import requests

# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True)
# print(inputs['input_ids'].shape)
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

