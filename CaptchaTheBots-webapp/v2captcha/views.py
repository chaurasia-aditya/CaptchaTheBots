from django.shortcuts import render
from .forms import CaptchaForm, UploadForm
from django.http import HttpResponse
import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import Post
from django.core.exceptions import ObjectDoesNotExist
from skimage.util import random_noise
import numpy as np
import cv2
from PIL import Image
import PIL.ImageOps
from io import BytesIO
import base64
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from art.attacks.evasion import HopSkipJump, ProjectedGradientDescent




# path = imgpath = str(settings.BASE_DIR) +"/templates/static/"+ "Bus/Bus (23).png"
# im = Image.open(path)
# im = im.convert('RGB')

# im = im.resize((150, 150))
# im_arr = np.array(im)
# im_arr = np.expand_dims(im_arr, axis=0)

# print(im_arr.shape)
# print(settings.MODEL.predict(im_arr).argmax())

# # print(f'Original Image Prediction: {settings.LABEL_NAMES[settings.MODEL.predict(im_arr).argmax()]}')

# print(settings.LABEL_DICT)



# Create your views here.
def index(request):
    if request.method == 'POST':
        form = CaptchaForm(request.POST)
        if form.is_valid():

            # Handle the form submission, e.g., save data to the database
            return render(request, 'v2captcha/success.html')
    else:
        form = CaptchaForm()
    return render(request, 'v2captcha/index.html', {'form': form})

def custom_captcha(request):

    imgpath = str(settings.BASE_DIR) + "/templates/static"
    items = os.listdir(imgpath)
    items.remove(".DS_Store")
    items.remove("Other")
    right = random.choice(items)

    rand = random.randint(3, 5)
    images = []
    for i in random.sample(os.listdir(imgpath + "/" + right), rand):
        images.append(right+"/" + i)

    for i in random.sample(os.listdir(imgpath + "/Other"), 9-rand):
        images.append("Other/" + i)

    random.shuffle(images)
    # print(images)
    noisy = []
    for i in images:
        path = imgpath = str(settings.BASE_DIR) +"/templates/static/"+ str(i)
        im = Image.open(path)
        im = im.convert('RGB')
        im = PIL.ImageOps.invert(im)
        im = im.resize((150, 150))
        im_arr = np.array(im)
        im_arr = np.expand_dims(im_arr, axis=0)
        
        label = settings.LABEL_DICT[i.split("/")[0]]
        rand = random.randint(0, 2)
        if rand == 0:
            noisy_img = generate_fgsm(im_arr,label,0.2).numpy()

        if rand == 1:
            noisy_img = im_arr + settings.HSJ_ARR
        if rand == 2:
            noisy_img = im_arr + settings.PGD_ARR
        
        noisy.append(image_to_base64(noisy_img))

    return render(request, 'v2captcha/login.html', {'n':9, 'images':images, 'right':right,'noisy': noisy, 'show':False})

def generate_fgsm(image, label, epsilon=0.1):
  image = tf.cast(image, tf.float32)

  with tf.GradientTape() as tape:
    tape.watch(image)
    prediction = settings.MODEL(image)
    loss = tf.keras.losses.MSE(label, prediction)
  gradient = tape.gradient(loss, image)
  sign_grad = tf.sign(gradient)
  perturbation = epsilon * sign_grad  # Apply epsilon to the sign gradient

  result = image + (perturbation * 0.05)
  return result

def generate_hopskipjump(target_image, classifier):
    target_image = target_image.astype(np.float32)
    attack = HopSkipJump(classifier=classifier, targeted=False, max_iter=0, max_eval=1000, init_eval=10)
    iter_step = 10
    x_adv = None
    for i in range(20):
        x_adv = attack.generate(x=np.array([target_image]), x_adv_init=x_adv, resume=True)

        attack.max_iter = iter_step
    return x_adv

def generate_pgd(X_test, classifier):
    pgd = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=.1, eps_step=0.1, max_iter=1,
                               targeted=False, num_random_init=0, batch_size=1000)
    x_test_adv = pgd.generate(X_test)
    return x_test_adv



def upload_image(request):

    try:
        if request.method == "POST":
            form = UploadForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()
            
            post = Post.objects.latest('id')
            noisy = add_noise()

            return render(request=request, template_name="v2captcha/upload_image.html", context={'form':form, 'post':post, 'noisy':noisy})
        
        form = UploadForm()
        post = Post.objects.latest('id')
        noisy = add_noise()

        # print(post.values())
        return render(request=request, template_name="v2captcha/upload_image.html", context={'form':form, 'post':None})
    except ObjectDoesNotExist:
        return render(request=request, template_name="v2captcha/upload_image.html", context={'form':form, 'post':None})


def add_noise():
    
    noisy_images = []
    post = Post.objects.latest('id')

    imgpath = str(settings.BASE_DIR) + str(post.image.url)

    im = Image.open(imgpath)
    im_arr = np.array(im)


    noise_factor = 0.02


    noise_img = random_noise(im_arr, mode="s&p",clip=True, amount=noise_factor)

    noisy_images.append(["S&P", image_to_base64(noise_img)])

    noise_img = random_noise(im_arr, mode='Poisson', clip=True)

    noisy_images.append(["Poisson", image_to_base64(noise_img)])

    noise_img = random_noise(im_arr, mode='gaussian', clip=True)

    noisy_images.append(["Gaussian", image_to_base64(noise_img)])

    noise_img = random_noise(im_arr, mode='speckle', clip=True)

    noisy_images.append(["Speckle", image_to_base64(noise_img)])

    return noisy_images



def image_to_base64(image):
    image = np.squeeze(image)
    image = Image.fromarray((image * 255).astype(np.uint8))
    buff = BytesIO()
    image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    img_str = img_str.decode("utf-8")  # convert to str and cut b'' chars
    return img_str
