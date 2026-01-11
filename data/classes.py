# Cityscapes 19个语义类别
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 
    'bus', 'train', 'motorcycle', 'bicycle'
]

# 类别ID映射 (34类 -> 19类)
ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 16, 31: 17, 32: 18, 33: 255, -1: 255
}

TRAINID_TO_COLOR = {
    0: (128, 64, 128),      # road
    1: (244, 35, 232),      # sidewalk
    2: (70, 70, 70),        # building
    3: (102, 102, 156),     # wall
    4: (190, 153, 153),     # fence
    5: (153, 153, 153),     # pole
    6: (250, 170, 30),      # traffic light
    7: (220, 220, 0),       # traffic sign
    8: (107, 142, 35),      # vegetation
    9: (152, 251, 152),     # terrain
    10: (70, 130, 180),     # sky
    11: (220, 20, 60),      # person
    12: (255, 0, 0),        # rider
    13: (0, 0, 142),        # car
    14: (0, 0, 70),         # truck
    15: (0, 60, 100),       # bus
    16: (0, 80, 100),       # train
    17: (0, 0, 230),        # motorcycle
    18: (119, 11, 32),      # bicycle
    255: (0, 0, 0)          # ignore
}