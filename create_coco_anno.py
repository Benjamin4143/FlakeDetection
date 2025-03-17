from labelme2coco.labelme2coco import get_coco_from_labelme_folder
import json
import os

###########################################################################
################## CHOOSE YOUR INPUTS BELOW HERE  ###########################

#choose here the path of the folder that contains your labelme .json files
#relative to the directory from which you are running this python file
labelme_folder = "path/to/labelme"

#choose here the relative path of the output .json file 
output_file = 'path/to/annotations.json'


#choose the mapping between category ids and category names here
coco_category_list = [
    {"id": 1, "name": "Mono"},
    {"id": 2, "name": "Few"},
    {"id": 3, "name": "Thick"}
]


################# DON'T CHANGE ANYTHING BELOW HERE #######################
#########################################################################

coco = get_coco_from_labelme_folder(labelme_folder, coco_category_list=coco_category_list)
coco_dict = coco.json
for image in coco_dict["images"]:
    image["file_name"] = os.path.basename(image["file_name"])
with open(output_file,'w') as file:
    json.dump(coco_dict, file, indent=4)

