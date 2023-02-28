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
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import tostring
import lxml.etree as etree

# codigo

#------ Funcoes de Apoio

#label permanece na imagem quando houver alguma alteração rotacional
def pad(image, by): 
    image_border1 = ia.pad(image, top=1, right=1, bottom=1, left=1,
                mode="constant", cval=255)
    image_border2 = ia.pad(image_border1, top=by-1, right=by-1,
                bottom=by-1, left
                =by-1,
                mode="constant", cval=0)
    return image_border2    

#label permanece na imagem quando houver alguma alteração rotacional
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

# verifica se a paleta é rgb
def corrige_RGB(data):
    im = Image.open(data)
    image_filename = data
    if not im.mode == 'RGB':
        print('NOT RGB')
        im = im.convert('RGB')
        byteIO = io.BytesIO()
        im.save(byteIO, format='PNG')
        byteArr = byteIO.getvalue()
        img = imageio.imread(byteArr)
        imageio.imwrite(data, img)

#--------- Inicialização

input_img = r'C:/Users/vitor.franca/projects/tecban/dataset/malas/images/'
filenames = os.listdir(input_img)
folder = 'replicadas' 

output_img = "{0}{1}/".format(input_img, folder)

output_xml = 'C:/Users/vitor.franca/projects/tecban/dataset/malas/label/replicadas_xml/'
input_dir = r'C:/Users/vitor.franca/projects/tecban/dataset/malas/label/'
# output_dir = r'C:/Users/vitor.franca/projects/tecban/imagens_replicadas/'

# img = glob(input_img + '*.jpg')
XML_files = glob(input_dir + '*.xml')

# print(XML_files) # XML_filenames  Essa variavel imprime a lista do ditorio e arquivo e extensão xml XML_filenames 
 
def main():
    indice = 0

    seq = iaa.Sequential([
        
        iaa.Affine(rotate=(-5, 5), mode='edge'),
        iaa.Affine(shear=(-5, 5), mode='edge'), 
        iaa.GaussianBlur(sigma=(0, 0.5)), 

        # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)), 
        # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),         
            # iaa.GaussianBlur(sigma=(0.0, 3.0)),
            # iaa.Fliplr(0.1),
            # iaa.Flipud(0.5),
            # iaa.Multiply((0.5, 1.5)),
            # iaa.Sharpen(alpha=0.5), 
            # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),  
            # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            # iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
        # iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1), per_channel=True),
        # iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))

        iaa.OneOf([
        iaa.AdditiveGaussianNoise(scale=0.02*255),
        iaa.Add(50, per_channel=True),     
        iaa.Multiply(1.0, per_channel=0.5),      
        ]), 

        iaa.OneOf([
        iaa.Sharpen(alpha=1, lightness=(1.0, 1.5)),    
        iaa.Emboss(alpha=1, strength=(0.5, 1.0)),    
        ])

    ])   
    
    for filename in XML_files: 
        filename = filename.replace('\\','/').replace(chr(92), '/') # for substituiu as duas barras invertidas no final do diretório
        
        print("Lendo Arquivo", filename)
        path, xml_file_name = os.path.split(filename) # ? 

        output = '{0}{1}.{2}'

        doc = minidom.parse(filename) 
        img_paths = doc.getElementsByTagName('path')
        image_filename = img_paths[0].firstChild.data
        image_filename = image_filename.replace('\\','/').replace(chr(92), '/')

        print(image_filename)
        #-----------------------------------------------------------------
        corrige_RGB(image_filename)

        print('------------>', image_filename)  
        path, img_file_name = os.path.split(image_filename)
        img_file_name, ext = img_file_name.split('.')
        
        img = imageio.imread(image_filename)    
        
        bndboxes = doc.getElementsByTagName('object')
        
        bounding_boxes = [] 
        boxes = []
        for bndbox in bndboxes:
            xmin = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
            label = doc.getElementsByTagName('name')[0].firstChild.data
            
            xname = doc.getElementsByTagName('name')[0].firstChild.data
            xpose = doc.getElementsByTagName('pose')[0].firstChild.data
            xtruncated = doc.getElementsByTagName('truncated')[0].firstChild.data
            xdifficult = doc.getElementsByTagName('difficult')[0].firstChild.data
            boxes.append({"xmin": xmin, "xmax": xmax, "ymin": ymin , "ymax": ymax, 
                          "name": xname, "pose": xpose, "truncated": xtruncated,
                          "difficult": xdifficult})
            
            bounding_boxes.append(BoundingBox(x1 = xmin, x2 = xmax, y1 = ymin , y2 = ymax, label = label)) 
 
        bbs = BoundingBoxesOnImage(bounding_boxes, shape=img.shape)        
        
        width = img.shape[0]
        height = img.shape[1]
        depth = img.shape[2]

        # modifying_img = seq.augment_image(img)
        # #ia.imshow(modifying_img) 
        # print("---> Gravando ", output.format(output_img, name,i,ext))
        # imageio.imwrite(output.format(output_img, name,i,ext), modifying_img)
        # print("finished convert L")
        # i=i+1
            
        i = indice
        print('------------>Gravando arquivo')
        print('               ',output.format(output_img, i,ext))
        imageio.imwrite(output.format(output_img, i,ext), img)
        
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = folder
        ET.SubElement(annotation, "filename").text = '{}.{}'.format(i,ext)
        ET.SubElement(annotation, "path").text = output.format(output_img, i,ext)
        ET.SubElement(annotation, "segmented").text = "0"

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)

        for box in boxes:
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = box["name"]
            ET.SubElement(object, "pose").text = box["pose"]
            ET.SubElement(object, "truncated").text = str(box["truncated"])
            ET.SubElement(object, "difficult").text = str(box["difficult"])
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(box["xmin"])
            ET.SubElement(bndbox, "xmax").text = str(box["xmax"])
            ET.SubElement(bndbox, "ymin").text = str(box["ymin"])
            ET.SubElement(bndbox, "ymax").text = str(box["ymax"])
            
        tree = ET.ElementTree(annotation)
        
        print('               ',output.format(output_xml, i,'xml'))
        tree.write(output.format(output_xml, i,'xml') )
        
        '''
        annotation = {'folder': folder, 
                     'filename': '{}.{}'.format(i,ext),
                     'path': output.format(output_img, i,ext), 
                     'source' : {'database' : 'Unknown'}, 
                     'size' : {'width': width, 'height': height, 'depth': depth}, 
                     'segmented': '0',  
                     'object': {
                                'name': xml_name,
                                'pose': xml_pose,
                                'truncated': xml_truncated,
                                'difficult' : xml_difficult,
                                'bndbox': boxes[0]
                         }
                    }
        xml = dicttoxml(annotation, custom_root='annotation', attr_type=False)  
        print('                             ',output.format(output_img, i,'xml'))
        with open(output.format(output_img, i,'xml'), 'w') as file:
            file.write(xml.decode("utf-8") )
         '''        
        
        #Codigo OK sem box
        #images = [img, img, img, img, img, img, img, img, img]
        #image_auge = seq.augment_images(images)
        #ia.imshow(ia.draw_grid(image_auge, cols=3, rows=4)) 
        #for j in range(0, len(images)):
        #    i = i + 1
        #    imageio.imwrite(output.format(output_img, i,ext), image_auge[j]) 
 

        for j in range(0, 15):
            image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
            image_after = draw_bbs(image_aug, bbs_aug.remove_out_of_image().clip_out_of_image(), 100)
            
            i = i + 1
            print('               ',output.format(output_img, i,ext))
            
            imageio.imwrite(output.format(output_img, i,ext), image_aug)
        
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "folder").text = folder
            ET.SubElement(annotation, "filename").text = '{}.{}'.format(i,ext)
            ET.SubElement(annotation, "path").text = output.format(output_img, i,ext)
            ET.SubElement(annotation, "segmented").text = "0"

            source = ET.SubElement(annotation, "source")
            ET.SubElement(source, "database").text = "Unknown"

            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(image_aug.shape[0])
            ET.SubElement(size, "height").text = str(image_aug.shape[1])
            ET.SubElement(size, "depth").text = str(image_aug.shape[2])

            j = 0
            for box in bbs_aug.bounding_boxes:
                object = ET.SubElement(annotation, "object")
                ET.SubElement(object, "name").text = boxes[j]["name"]
                ET.SubElement(object, "pose").text = boxes[j]["pose"]
                ET.SubElement(object, "truncated").text = str(boxes[j]["truncated"])
                ET.SubElement(object, "difficult").text = str(boxes[j]["difficult"])
                bndbox = ET.SubElement(object, "bndbox")

                xmin = int(box.x1)
                ymin = int(box.y1)
                xmax = int(box.x2)
                ymax = int(box.y2)

                '''
                caso os valores da tag bndbox seja negativa esse função 
                transaforma em zero para x e para y permanesse o valor maior
                '''

                if xmin < 0:
                    xmin = 0

                if ymin < 0:
                    ymin = 0 

                if xmax < 0: 
                    xmax = width


                if xmax > width: 
                    xmax = width

                if ymax < 0:
                    ymax = height 

                if ymax > height:
                    ymax = height 


                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "ymax").text = str(ymax) 

                # ET.SubElement(bndbox, "xmin").text = str(box.x1)
                # ET.SubElement(bndbox, "xmax").text = str(box.x2)
                # ET.SubElement(bndbox, "ymin").text = str(box.y1)
                # ET.SubElement(bndbox, "ymax").text = str(box.y2)
                j = j + 1


            tree = ET.ElementTree(annotation)

            print('               ',output.format(output_xml, i,'xml'))
            # etree.tostring(output.format(output_xml, i,'xml'), pretty_print=True)
            tree.write(output.format(output_xml, i,'xml'))

        indice = i + 1

        print('OK')
        # break

if __name__ == "__main__":  
     main()     