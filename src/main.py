from blood_pressure import extract_blood_pressure
from deshadow import deshadow_and_normalize_image
from PIL import Image

image_filepath = "C:/Users/vcz2aj/Documents/jupyter_notebooks/Rwandan_30_200_retraining/datasets/300_20/images/train/BP_section_of_17FullTestI-1DrChristian9Intraoperative.jpg"
dest_filepath = "C:/Users/vcz2aj/Pictures/"
test_image = Image.open(image_filepath)
normed_image = deshadow_and_normalize_image(test_image)
test_image.save(dest_filepath + "test.jpg")
print(extract_blood_pressure(test_image))
