def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return(int((x1 + x2) / 2), int((y1 + y2) / 2))

def interpolate_points(p1, p2, alpha):
    if p1[0] is None or p2[0] is None:
        return (None, None) 
    x = p1[0] * (1 - alpha) + p2[0] * alpha
    y = p1[1] * (1 - alpha) + p2[1] * alpha
    return (x, y)

def bbox_feet(bbox):
    x1, _, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))