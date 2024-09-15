def get_box_center(bounding_box):
    #returns center point of bounding box
    x1,y1,x2,y2= bounding_box
    return int((x2+x1)//2),int((y2+y1)//2)

def get_box_width(bounding_box):
    #index 2 is x2 and index 1 is x1
    return bounding_box[2]-bounding_box[0]