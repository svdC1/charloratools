# charloratools 

![Alt text](https://res.cloudinary.com/ducarqe1h/image/upload/v1726127296/clt_banner_w5hpan.png)

This is a simple python library which intends to **facilitate the process of image gathering,filtering and processing when creating a training dataset to use for training stable diffusion LoRas**. As the name suggests it is specifically useful when organizing a *character lora* image training dataset.






## Features

### Image Scraping

Provides tools for scraping **Instagram**,**VSCO** and **X** profiles using **selenium** and **webdriver manager**.

- **LOGIN WARNING** : If you don't want to login when using the scraping tools it is possible by providing invalid or empty credentials when instantiating any of the scraper classes, but without providing valid credentials most of the times the scraping fails or is able to scrape only 3 or 4 images.


- **WARNING WHEN USING LOGIN** : The scraping functions are very simple and don't use any APIs, they simply start a selenium webdriver to automatically login, scroll through the provided profile, gather all image sources and download them using the requests library, when testing on *VSCO* and *X* I didn't notice any warnings or account problems but use it while logging in **at your own risk**.

- As for **Instagram** I didn't use instaloader as there seems to be a bug when using it with python 3.10.6 so the scraping works just like the other scraper classes, with that said *Instagram has a more restrictive policy for account automation* so I don't recommend the logged in use with a headless webdriver as it is easily detected.*See the example usage for using the scrapers without headless mode*.

### Image Filtering

Provides an easy usage of **facenet_pytorch** implementation of *MTCNN* and *IncepetionResnetV1* for detecting and recognizing faces in images.

- Select all images in which faces ared detected and copy them to a different directory.

- Select all images in which only one face is detected and copy them to a different directory

- Select all images in which a specific face is recognized and copy them to a different directory, this is the function which uses the *IncepetionResnetV1* facenet_pytorch implementation and defaults to using the model pre-trained on *vggface2* dataset for embedding generation.

- You can customize the functions by changing the keyword arguments explained in the docstring and documentation

### Image Directory Management

Provides easy image directory management with the *GalleryManager* and *ImgManager* classes which implements simple syntax for managing images between directories using overloading of python's built-in operators. They require you to specify which **hashing algorithm** you want to use for producing image hashes which the classes uses when assessing if images are equal or not, All hashes, except for *sha256*, use the **ImageHash** library implementation. The options are : **[*phash*, *average_hash*,*crop_resistant_hash* (using *dhash*) and *dhash*]**

## Installation

[Install with pip](https://pypi.org/project/charloratools/)

```bash
pip install charloratools
```

**Important : Install Torch according to your system requirements, considerations:**

- **When building, pip will try to verify the CUDA version installed in your system, but if that fails or you have an AMD GPU you'll want to install the correct torch distribution.**

- **When pip can't figure out the CUDA version, it will install the CPU only distribution of torch**

- **If pip can't figure out the CUDA version, you'll get a warning during installation that "facenet_pytorch" one of the required libraries wasn't installed that is because facenet_pytorch requires torch to function properly and the CPU-only version is too slow for the models here used. But if you want to use the CPU only version anyway all you have to do is install facenet_pytorch==2.6.0 using pip.**


**For NVIDIA GPUs these are the commands for each CUDA version**

- Cuda 11.8
```bash
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Cuda 12.1
```bash
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage/Examples

### Scrapers
All of the scraper classes are written as context managers

```python
import charloratools

vscos=charloratools.Scrapers.VSCOScraper(email='me@example.com',password='password')

xs=charloratools.Scrapers.XScraper(username='X @',password='password')

instas=charloratools.InstagramScraper(username='instagram @',password='password')

with vscos as scraper:
    scraper.get_vsco_pics(username='@',save_path='usr/local')

with xs as scraper:
    scraper.get_x_pics(username='@',save_path='usr/local')

with instas as scraper:
    scraper.get_feed_pics(username='@',save_path='usr/local')

```

### Filters

- The filter functions return a new *GalleryManager* object for the new directory
```python
import charloratools

FR=charloratools.FilterAI.FaceRecognizer(path='usr/local')

FR.filter_images_without_face(output_dir='filtered')

FR.filter_images_with_multiple_faces(output_dir='filtered2')

FR.filter_images_without_specific_face(ref_img_path='ref_img.jpg',output_dir='filtered3')

```


### Image Directory Management

- Initializing classes

```python
import charloratools

gallery1=charloratools.SysFileManager.GalleryManager(path='usr/local/dir1',hashtype='sha256')

gallery2=charloratools.SysFileManager.GalleryManager(path='usr/local/dir2',hashtype='phash')

single_img=charloratools.SysFileManager.ImgManager(path='usr/local/test.jpg',hashtype='sha256')
```
- When performing operations on instances with different hash types the right-hand instance's hash type is converted to match the left-hand instance's.
```python
len(gallery1) # Number of images in dir

gallery1==gallery2 # individual image hash comparison
```
- Operations for copying and creating new directory
```python
gallery1+gallery2
gallery1-gallery2
```
- Same Operations but inplace

```python
gallery1+=gallery2
gallery1-=gallery2

```
### Extras for Image Visualization

Creating an HTML image gallery file with inline CSS and JS (Uses fotoroma.io),Images are converted to base64 html img tags

```python
import charloratools

gallery=charloratools.SysFileManager.GalleryManager(path='usr/local/imgdir')
gallery.to_html_img_gallery(output_dir='usr/local')
```

### [Available as ComfyUI Custom Node](https://github.com/svdC1/comfy-ui-lora-dataset-tools)



## Roadmap

- Create more detailed documentation
