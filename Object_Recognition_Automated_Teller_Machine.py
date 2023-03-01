from PIL import Image
from PIL import ImageFilter
import numpy as np
import os
import imgaug as ia
import imgaug.augmenters as iaa
import cv2

input_dir = r'/home/vitor/Documents/y/'
filenames = os.listdir(input_dir)

output_dir = r'/home/vitor/Documents/x/'

#images = []

def main():
     
    for filename in filenames:
        i = 0
        name, ext = filename.split('.')
        output = '{0}{1}_{2}.{3}'

        img = Image.open(input_dir + filename)

        img0 = img.convert("L")
        img0.save(output.format(output_dir, name,i,ext))
        print("finished convert L")
        i=i+1

        # img1 = img0.convert("1")
        # img1.save(output.format(output_dir, name,i,ext))
        # print("finished convert 1")
        # i=i+1
        
        # img2 = img0.filter(ImageFilter.BLUR(2))
        # img2.save(output.format(output_dir, name,i,ext))
        # print("ImageFilter.BLUR")
        # i=i+1

        #img3 = Image.open(input_dir + filename)
        #img3 = img.filter(ImageFilter.CONTOUR)
        #img3.save(output.format(output_dir, name,i,ext))
        #print("ImageFilter.CONTOUR")
        #i=i+1
        
        img4 = img0.filter(ImageFilter.DETAIL)
        img4.save(output.format(output_dir, name,i,ext))
        print("ImageFilter.DETAIL")
        i=i+1
        
        img5 = img0.filter(ImageFilter.EDGE_ENHANCE)
        img5.save(output.format(output_dir, name,i,ext))
        print("ImageFilter.EDGE_ENHANCE")
        i=i+1  
        
        img6 = img0.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img6.save(output.format(output_dir, name,i,ext))
        print("ImageFilter.EDGE_ENHANCE_MORE")
        i=i+1   

        #img7 = Image.open(input_dir + filename)
        #img7.save(output.format(output_dir, name,i,ext))
        #print("ImageFilter.EMBOSS")
        #i=i+1

        #img8 = Image.open(input_dir + filename)
        #img8 = img.filter(ImageFilter.FIND_EDGES)
        #img8.save(output.format(output_dir, name,i,ext))
        #print("ImageFilter.FIND_EDGES")
        #i=i+1
        
        img9 = img0.filter(ImageFilter.SMOOTH)
        img9.save(output.format(output_dir, name,i,ext))
        print("ImageFilter.SMOOTH")
        i=i+1
       
        img10 = img0.filter(ImageFilter.SMOOTH_MORE)
        img10.save(output.format(output_dir, name,i,ext))
        print("ImageFilter.SMOOTH_MORE")
        i=i+1
        
        img11 = img0.filter(ImageFilter.SHARPEN)
        img11.save(output.format(output_dir, name,i,ext))
        print("ImageFilter.SHARPEN")
        i=i+1   

        img12 = img0.filter(ImageFilter.GaussianBlur(radius = 1))
        img12.save(output.format(output_dir, name,i,ext))
        print("ImageFilter.GaussianBlur(radius = 9)")
        i=i+1         
        

if __name__ == "__main__": 

    main()
    print("finished") 

