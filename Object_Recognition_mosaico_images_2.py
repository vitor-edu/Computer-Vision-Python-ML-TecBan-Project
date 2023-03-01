from imgaug import augmenters as iaa
import numpy as np
import imgaug as ia
import imgaug.parameters as iap
import os
import io 
import imageio
from PIL import Image
from xml.dom import minidom 
from glob import glob   
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import matplotlib.pyplot as plt
import imghdr 
from scipy import misc


#------ Funcoes de Apoio
def pad(image, by): #label permaneça na imagem quando houver alguma alteração rotacional
    image_border1 = ia.pad(image, top=1, right=1, bottom=1, left=1,
                mode="constant", cval=255)
    image_border2 = ia.pad(image_border1, top=by-1, right=by-1,
                bottom=by-1, left
                =by-1,
                mode="constant", cval=0)
    return image_border2            

def draw_bbs(image, bbs, border):
    GREEN = [0, 255, 0]
    ORANGE = [255, 140, 0]
    RED = [255, 0, 0]
    image_border = pad(image, border)
    for bb in bbs.bounding_boxes:
        if bb.is_fully_within_image(image.shape):
            color = GREEN
        elif bb.is_partly_within_image(image.shape):
            color = ORANGE
        else:
            color = RED
        image_border = bb.shift(left=border, top=border)\
                    .draw_on_image(image_border, size=2,
    color=color)
    return image_border

#--------- Inicialização
input_img = r'C:/Users/vitor.franca/projects/tecban/dataset/armas_de_fogo/images/'
filenames = os.listdir(input_img)
output_img = r'C:/Users/vitor.franca/projects/tecban/imagens_replicadas/'

input_dir = r'C:/Users/vitor.franca/projects/tecban/dataset/armas_de_fogo/labels/'
# output_dir = r'C:/Users/vitor.franca/projects/tecban/imagens_replicadas/'

# img = glob(input_img + '*.jpg')
XML_files = glob(input_dir + '*.xml')

indice = 0
 
def main():
    for filename in XML_files: 
        filename = filename.replace('\\','/').replace(chr(92), '/')

        print("Lendo Arquivo", filename)
        path, xml_file_name = os.path.split(filename)
        
        output = '{0}{1}.{2}'

        doc = minidom.parse(filename) 
        bndboxes = doc.getElementsByTagName('bndbox')
        img_paths = doc.getElementsByTagName('path')

        for img_path in img_paths:
            data = img_paths[0].firstChild.data
            im = Image.open(data)

            image_filename = data
            if not im.mode == 'RGB':
                print('NOT RGB')
                im = im.convert('RGB')
                byteIO = io.BytesIO()
                im.save(byteIO, format='PNG')
                byteArr = byteIO.getvalue()
                img = imageio.imread(byteArr)
            else:
                img = imageio.imread(data)  

        print('------------>', image_filename)  
        path, img_file_name = os.path.split(image_filename)
        img_file_name, ext = img_file_name.split('.')
        
        bounding_boxes = [] 
        for bndbox in bndboxes:
            xmin = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)

            bounding_boxes.append(BoundingBox(x1 = xmin, x2 = xmax, y1 = ymin , y2 = ymax)) 

        bbs = BoundingBoxesOnImage([bounding_boxes], shape=img.shape)

        #img = imageio.imread(input_img + filename)  
        img = imageio.imread(img_paths)

        seq = iaa.Sequential([
            iaa.Affine(rotate=(-45, 45), shear=(-8, 8)),        
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.5, 1.5)),
            ])   

        modifying_img = seq.augment_image(img)
        #ia.imshow(modifying_img) 
        print("---> Gravando ", output.format(output_img, name,i,ext))
        imageio.imwrite(output.format(output_img, name,i,ext), modifying_img)
        print("finished convert L")
        i=i+1
         
        images = [img, img, img, img, img, img, img, img, img]
        image_auge = seq.augment_images(images)
        ia.imshow(ia.draw_grid(image_auge, cols=3, rows=3)) 

        for j in range(0, len(images)):
           imageio.imwrite(output.format(output_img, name,i,ext), image_auge[j])

        i = indice
        print('------------>Gravando arquivo')
        print('                             ',output.format(output_img, i,ext))
        #imageio.imwrite(output.format(output_img, i,ext), img)





#         ia.seed(1)

#         seq = iaa.Sequential([
#             iaa.GammaContrast(1.5),
#             iaa.Affine(translate_percent={"x": 0.1}, scale=0.8)
#         ])

#         image_aug, bbs_aug = seq(image=[img, img, img, img, img, img, img, img, img], bounding_boxes=bbs)

# #     im = imageio.imread(img[0])
# #     image_aug, bbs_aug = seq(image=im, bounding_boxes=bbs)
# #     # image_after = draw_bbs(image_aug, bbs_aug.remove_out_of_image().clip_out_of_image(), 30)
# #     ia.imshow(image_aug) 




#         ia.imshow(image_aug) 






#         #for image in images:
#         #    i = i + 1
#         #    image_auge = seq.augment_images(image)

#         #    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
#         #    image_after = draw_bbs(image_aug, bbs_aug.remove_out_of_image().clip_out_of_image(), 30)

#         #    imageio.imwrite(output.format(output_img, i,ext), image_auge)

            
#         break

# if __name__ == "__main__":  
#      main()    

#     for img_path in img_paths:
#         data = img_paths[0].firstChild.data
#         im = Image.open(data)

#     im = imageio.imread(img[0])
#     image_aug, bbs_aug = seq(image=im, bounding_boxes=bbs)
#     # image_after = draw_bbs(image_aug, bbs_aug.remove_out_of_image().clip_out_of_image(), 30)
#     ia.imshow(image_aug) 





# new_list = [] 
    
# def main():

#     for x in img: 

#         i = 0
#         name, ext = x.split('.')
#         output = '{0}{1}_{2}.{3}'        
#         filename_img = Image.open(input_dir2 + x)

#         img0 = filename_img.convert("L")

#         img0.save(output.format(output_dir2, name,i,ext))
#         # print("finished convert L")
#         # print(filename_img)    
            #  i=i+1                   


 #         # for filename in filenames:
#         #     name, ext = filename.split('.')
#         #     output = '{0}{1}_{2}.{3}'

#         #     output_dir.save(filename(output.format(output_dir, name,i,ext)))  
# 
#                                    
            

#         # print(bounding_boxes)  #gerou a localização posicional do label  
#         # print(img_paths) #printou a tag path [<DOM Element: path at 0x2764fc1d228>]        
#         # image_aug, bbs_aug = move(image=img_paths, boundingS_boxes=bounding_boxes)
#         # ia.imshow(bbs_aug.draw_on_image(image_aug, size=5)) 
        
# new_list.append(x)
# # # print(new_list)  

# if __name__ == "__main__":  
#      main()             
# 
# 
# 
# 
# 
# 
# 
# 
#                  

#                 seq = iaa.Sequential([

#                     iaa.Affine(rotate=(-45, 45), shear=(-8, 8)),        
#                     iaa.GaussianBlur(sigma=(0.0, 3.0)),
#                     iaa.Fliplr(0.5),
#                     iaa.Flipud(0.5),
#                     iaa.Multiply((0.5, 1.5)), 
                    
#                     iaa.SomeOf(2, [
#                         iaa.Affine(rotate=45),
#                         iaa.AdditiveGaussianNoise(scale=0.2*255),
#                         iaa.Add(50, per_channel=True),
#                         iaa.Sharpen(alpha=0.5)
#                     ], random_order=True), 

#                     iaa.OneOf([
#                     iaa.Affine(rotate=45),
#                     iaa.AdditiveGaussianNoise(scale=0.2*255),
#                     iaa.Add(50, per_channel=True),
#                     iaa.Sharpen(alpha=0.5), 
#                     iaa.WithChannels(0, iaa.Affine(rotate=(0, 45)))
#                     ]),  

#                     iaa.SomeOf((0, None), [
#                     iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1)),     
#                     iaa.Affine(rotate=45),
#                     iaa.AdditiveGaussianNoise(scale=0.2*255),
#                     iaa.Add(50, per_channel=True),
#                     iaa.Sharpen(alpha=0.5)
#                     ], random_order=True),                                   

#                 ], random_order=True)                     


#         # for filename in filenames:

#         #     name, ext = filename.split('.')
#         #     output = '{0}{1}_{2}.{3}'

#         #     output_dir.save(filename(output.format(output_dir, name,i,ext)))     





#         # seq = iaa.Sequential([

        #     iaa.Affine(rotate=(-45, 45), shear=(-8, 8)),        
        #     iaa.GaussianBlur(sigma=(0.0, 3.0)),
        #     iaa.Fliplr(0.5),
        #     iaa.Flipud(0.5),
        #     iaa.Multiply((0.5, 1.5)), 
            
        #     iaa.SomeOf(2, [
        #         iaa.Affine(rotate=45),
        #         iaa.AdditiveGaussianNoise(scale=0.2*255),
        #         iaa.Add(50, per_channel=True),
        #         iaa.Sharpen(alpha=0.5)
        #     ], random_order=True), 

        #     iaa.OneOf([
        #     iaa.Affine(rotate=45),
        #     iaa.AdditiveGaussianNoise(scale=0.2*255),
        #     iaa.Add(50, per_channel=True),
        #     iaa.Sharpen(alpha=0.5), 
        #     iaa.WithChannels(0, iaa.Affine(rotate=(0, 45)))
        #     ]),       

           
        #     iaa.SomeOf((0, None), [
        #     iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1)),     
        #     iaa.Affine(rotate=45),
        #     iaa.AdditiveGaussianNoise(scale=0.2*255),
        #     iaa.Add(50, per_channel=True),
        #     iaa.Sharpen(alpha=0.5)
        #     ], random_order=True),                                   

        # ], random_order=True)   

        

                        
    # name, ext = filename.split('.')
    #     output = '{0}{1}_{2}.{3}'

    #     img = Image.open(input_dir + filename)    
    #     img0.save(output.format(output_dir, name,i,ext))   
    #     i=i+1


    # iaa.ReplaceElementwise(
            # iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
            # iap.Normal(128, 0.4*128),
            # per_channel=0.5), 

            # iaa.Sequential([  
            # iaa.Affine(rotate=(-45, 45), shear=(-8, 8))        
            # ], random_order=True), 

            # iaa.Sequential([
            # iaa.Affine(rotate=(-25, 25), shear=(-8, 8)),
            # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),  
            # iaa.Fliplr(0.5), 
            # iaa.Alpha((0.0, 1.0), first=iaa.Add(100), second=iaa.Multiply(0.2))
            # ], random_order=True)                
    

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

#        



