import requests
from os.path import exists
from PIL import Image
#Color processing
from colorthief import ColorThief
import numpy as np

img_folder = 'offers_imgs/'

def download_img(img_url):
    img_path = url_to_path(img_url)
    response = requests.get(img_url)
    if not response.ok:
        return None
    img_data = response.content
    with open(img_path, 'wb') as handler:
        handler.write(img_data)
        return img_path
        
def url_to_path(url):
    end = url.index('?')
    start = end - 36
    return img_folder + url[start:end]

def get_img(url):
    path = url_to_path(url)
    if exists(path):
        return path
    else:
        return download_img(url)

def get_concat_h(img_url_list):
    im = []
    for url in img_url_list:
        imgurl = get_img(url)
        im += [Image.open(get_img(url))] if imgurl is not None else []
    tot_width = sum([el.width for el in im])
    dst = Image.new('RGB', (tot_width, im[0].height))
    shift = 0
    for el in im:
        dst.paste(el, (shift, 0))
        shift += el.width
    return dst

class MyColorThief(ColorThief):
    def __init__(self, img):
        self.image = img
        
def get_palette_imgs(img_list, color_count=4):
    return MyColorThief(get_concat_h(img_list)).get_palette(color_count)

def color_difference(col_1, col_2):
    r_1, r_2 = col_1[0], col_2[0]
    g_1, g_2 = col_1[1], col_2[1]
    b_1, b_2 = col_1[2], col_2[2]
    return ((r_1-r_2)**2 + (g_1-g_2)**2 + (b_1-b_2)**2)**(1/2) / ((3*255**2)**(1/2))

def color_palette_most_similar(col, pal):
    score = 1
    most_similar = pal[0]
    for c in pal:
        diff = color_difference(col, c)
        if diff < score:
            score = diff
            most_similar = c
    return most_similar

def palette_distance(pal_1, pal_2):
    return sum([color_difference(col_1, color_palette_most_similar(col_1, pal_2)) 
                for col_1 in pal_1])/len(pal_1)

def palette_similarity(pal_1, pal_2):
    return 1 - palette_distance(pal_1, pal_2)

#pal = get_palette_imgs(img_list_u)
#palette_similarity(pal, [(72, 63, 49), (112, 96, 78), (225, 224, 217), (145, 121, 111)])