    import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug.parameters as iap
import os
import io 
import imageio
from PIL import Image
from xml.dom import minidom 
from glob import glob   
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
   
image = imageio.imread("/home/vitor/Documents/221.jpg")
ia.imshow(image)

#---------------------------------------------------------------------------------------

# bbs = BoundingBoxesOnImage([BoundingBox (x1 =  , x2 = , y1 = , y2 = )], shape = im1.shape)
#     ia.imshow(bbs.draw_on_image(image, size=2))  

# image = r'C:/Users/vitor.franca/projects/tecban/revolver_teste/26.jpg'
# filenames = os.listdir(image)

# output_dir = r'C:/Users/vitor.franca/projects/tecban/dataset_replicado/'

def main():   
#   #1
#         seq = iaa.Sequential([

#             iaa.Affine(rotate=(-25, 25), shear=(-8, 8)),
#             iaa.AverageBlur(k=((5, 11), (1, 3))),
#             iaa.MedianBlur(k=(3, 11)),
#             iaa.GaussianBlur(sigma=(0.0, 3.0)),
#             iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
#             iaa.MotionBlur(k=15), 
#             iaa.Fliplr(0.5),
#             iaa.Sharpen(alpha=0.5),    
#             # iaa.AssertShape((None, 32, 32, 3)),

#             iaa.Multiply((0.5, 1.5)),
#             iaa.Multiply((0.5, 1.5), per_channel=0.5),
#             iaa.MultiplyElementwise((0.5, 1.5))   
#         ])
#         images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]

#         image_auge = seq.augment_images(images)
        # ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

    #2
        seq = iaa.Sequential([
        
            iaa.Affine(rotate=(-45, 45), shear=(-8, 8)),
            iaa.AverageBlur(k=((5, 11), (1, 3))),
            iaa.MedianBlur(k=(3, 11)),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.MotionBlur(k=15), 

            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sharpen(alpha=0.5),       

            iaa.Multiply((0.5, 1.5)),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.MultiplyElementwise((0.5, 1.5))   
        ])
        images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]

        image_auge = seq.augment_images(images)
        ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #3
#         seq = iaa.Sequential([
        
#             iaa.Fliplr(0.1),
#             iaa.Flipud(0.5),
#             iaa.Sharpen(alpha=0.3),         
#         ])
#         images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))
#     #4
#         seq = iaa.Sequential([

#             iaa.Affine(rotate=(-45, 45), shear=(-8, 8)),        
#             iaa.GaussianBlur(sigma=(0.0, 3.0)),
#             iaa.Fliplr(0.5),
#             iaa.Flipud(0.5),
#             iaa.Multiply((0.5, 1.5)),        
#         ])
#         images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #5
#         seq = iaa.Sequential([
        
#             iaa.Affine(rotate=(-45, 45), shear=(-8, 8)),
#             iaa.WithChannels(0, iaa.Affine(rotate=(0, 45))),    
#             iaa.Fliplr(0.5),
#             iaa.Flipud(0.5),
#             iaa.Multiply((0.5, 1.5)),        
#         ])
#         images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #6
#         seq = iaa.Sequential([     

#         iaa.Affine(rotate=(-45, 45), shear=(-8, 8))        
#         ])
#         images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #7
#         seq = iaa.Sequential([

#             iaa.Affine(rotate=(-25, 25), shear=(-8, 8)),
#             iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),  
#             iaa.Fliplr(0.5),
#         ])
#         images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #8
#         seq = iaa.Sequential([

#             iaa.ReplaceElementwise(
#             iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
#             iap.Normal(128, 0.4*128), per_channel=0.5), 
#             iaa.Affine(rotate=(-25, 25), shear=(-8, 8)),    
#         ]) 
#         images = [image for i in range(16)]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #9
#         seq = iaa.Sequential([

#         iaa.AddElementwise((-40, 40)), 
#         iaa.Affine(rotate=(-25, 25), shear=(-8, 8)), 
#         iaa.Fliplr(0.5),
#         iaa.Flipud(0.5),
#         ])

#         images = [image for i in range(16)]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #10
#         seq = iaa.Sequential([
#             iaa.GammaContrast((0.5, 2.0)),        
#             iaa.Affine(rotate=(-25, 25), shear=(-8, 8)),    
#         ])

#         images = [image for i in range(16)]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #11
#         seq = iaa.Sequential([  
#              iaa.PiecewiseAffine(scale=(0.01, 0.05)), 
#              iaa.Affine(rotate=(-25, 25), shear=(-8, 8)),    
#         ])

#         images = [image for i in range(16)]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #12

#         seq = iaa.Sequential([   
#              iaa.PerspectiveTransform(scale=(0.01, 0.15)), 
#              iaa.Fliplr(0.5),
#              iaa.Flipud(0.5),           
#         ])

#         images = [image for i in range(16)]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#     #13
#         seq = iaa.Sequential([

#             iaa.Affine(rotate=(-25, 25), shear=(-8, 8)),
#             iaa.AverageBlur(k=((5, 11), (1, 3))),
#             iaa.MedianBlur(k=(3, 11)),
#             iaa.GaussianBlur(sigma=(0.0, 3.0)),
#             iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
#             iaa.MotionBlur(k=15), 
#             iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)), 
#             iaa.Fliplr(0.5),
#             iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)), 
#             iaa.Sharpen(alpha=0.5),    
#             iaa.AssertShape((None, 32, 32, 3)),

#             iaa.Multiply((0.5, 1.5)),
#             iaa.Multiply((0.5, 1.5), per_channel=0.5),
#             iaa.MultiplyElementwise((0.5, 1.5))   
#         ])
#         images = [image, image, image, image, image, image, image, image, image, image, image, image, image, image, image, image]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

#         seq = iaa.Sequential([

#             iaa.AverageBlur(k=((5, 11), (1, 3))),
#             iaa.MedianBlur(k=(3, 11)),
#             iaa.GaussianBlur(sigma=(0.0, 3.0)),
#             iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
#             iaa.MotionBlur(k=15),      
#         ])
#         images = [image, image, image, image, image, image, image, 
#                  image, image, image, image, image, image, image, image, image]
#         image_auge = seq.augment_images(images)
#         ia.imshow(ia.draw_grid(image_auge, cols=4, rows=4))

if __name__ == "__main__": 

    main()
    print("finished") 


