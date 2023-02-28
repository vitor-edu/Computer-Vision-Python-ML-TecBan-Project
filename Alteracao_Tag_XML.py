import os
from glob import glob 
import imageio
from PIL import Image
import imgaug as ia
from natsort import natsorted
from xml.dom import minidom 

# altera a tag folder do xml, depois de alterado o caminho com o nome do arquivo.

input_path = r'C:/Users/vitor.franca/projects/tecban/repositorio2/'
input_glob = glob(input_path + '*.xml')
output_dir = r'C:/Users/vitor.franca/projects/tecban/repositorio/'
output_img = r'C:/Users/vitor.franca/projects/tecban/dataset/malas/images/'

for path in natsorted(input_glob): 
    # print('path: ', path) # retorna o diretorio + arquivo + extensao
    basename = os.path.basename(path)                    
    xml_fname, xml_ext = os.path.splitext(basename)

    doc = minidom.parse(path)
    
    path_nodes = doc.getElementsByTagName('path')
    
    img_path = path_nodes[0].firstChild.data
    # img_dir = os.path.dirname(img_path)
    img_fname, img_ext = os.path.splitext(os.path.basename(img_path))     # retorna o diretorio arquivo e extensao jpg
    # print(img_fname)
    # print(img_ext)
    new_img_path = os.path.join(output_img, img_fname + img_ext)
    print(new_img_path)
    path_nodes[0].firstChild.replaceWholeText(new_img_path)  

    output_path = os.path.join(output_dir, img_fname + img_ext)

    folder_nodes = doc.getElementsByTagName('folder')
    filename = folder_nodes[0].firstChild.data
    img_fname, img_ext = os.path.splitext(filename)   #retorna o arquivo e extensão jpg 

    new_folder_nodes = 'images'                       #variavel alterada na tag folder
    folder_nodes[0].firstChild.replaceWholeText(new_folder_nodes)
    # print('folder: ', new_folder_nodes)       
    
    output_path = os.path.join(output_dir, xml_fname + xml_ext)
    
    with open(output_path, 'w') as fh:
        xml_declaration = '<?xml version="1.0" ?>' # salvar arquivo xml 
        xml = doc.toxml()[len(xml_declaration):]
        fh.write(xml)