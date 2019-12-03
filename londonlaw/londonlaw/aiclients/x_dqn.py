import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
from torch import optim
import numpy as np

def one_hot_encode(value, size, max_value):
      a = np.array(value)
      b = np.zeros((size, max_value))
      b[np.arange(size), a-1] = 1
      return b.tolist()

def generate_feature_space(inital_st):
  #inital_st = [26, 5, [34, 29, 117, 174, 112], [[10, 8, 4], [10, 8, 4], [10, 8, 4], [10, 8, 4], [10, 8, 4]], 1]
  feature_space = []
  feature_space.extend(one_hot_encode(inital_st[0], 1, 199)[0])
  feature_space.append(inital_st[1])
  det_loc = one_hot_encode(inital_st[2], 5, 199)
  feature_space.extend(det_loc[0])
  feature_space.extend(det_loc[1])
  feature_space.extend(det_loc[2])
  feature_space.extend(det_loc[3])
  feature_space.extend(det_loc[4])
  det_tic = inital_st[3]
  feature_space.extend(det_tic[0])
  feature_space.extend(det_tic[1])
  feature_space.extend(det_tic[2])
  feature_space.extend(det_tic[3])
  feature_space.extend(det_tic[4])
  feature_space.append(inital_st[4])
  #print(feature_space)
  #print(len(feature_space))
  return feature_space


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.ll1 = nn.Linear(1211, 708)
        self.ll2 = nn.Linear(708, 708)
        self.ll3 = nn.Linear(708, 354)
        self.ll4 = nn.Linear(354, 354)
        self.oll = nn.Linear(354, 200) #changed from 16 to 200
    
    #function implements a forward pass to the network
    def forward(self, x):
        # x = x.flatten(start_dim=1) - flattening needed only for images
        x = F.relu(self.ll1(x))
        x = F.relu(self.ll2(x))
        x = F.relu(self.ll3(x))
        x = F.relu(self.ll4(x))
        x = F.softmax(self.oll(x))
        return x

def predict_best_move(state):
    TAXI = "TAXI"
    BUS = "BUS"
    UNDERGROUND = "UNDERGROUND"
    BLACK = "BLACK"
    board = (( (8, (TAXI,)), (9, (TAXI,)), (58, (BUS,)), (46, (BUS, UNDERGROUND)) ),  # locToRoutes[0] has no value; need to access locToRoutes[1] and higher               
            ( (8, (TAXI,)), (9, (TAXI,)), (58, (BUS,)), (46, (BUS, UNDERGROUND)) ),     # 001 
            ( (20, (TAXI,)), (10, (TAXI,)) ),
            ( (11, (TAXI,)), (12, (TAXI,)), (4, (TAXI,)), (22, (BUS,)), (23, (BUS,)) ),
            ( (3, (TAXI,)), (13, (TAXI,)) ),
            ( (15, (TAXI,)), (16, (TAXI,)) ),                                           # 005
            ( (29, (TAXI,)), (7, (TAXI,)) ),
            ( (6, (TAXI,)), (17, (TAXI,)), (42, (BUS,)) ),
            ( (1, (TAXI,)), (19, (TAXI,)), (18, (TAXI,)) ),
            ( (1, (TAXI,)), (19, (TAXI,)), (20, (TAXI,)) ),
            ( (2, (TAXI,)), (11, (TAXI,)), (34, (TAXI,)), (21, (TAXI,)) ),              # 010
            ( (3, (TAXI,)), (10, (TAXI,)), (22, (TAXI,)) ),
            ( (3, (TAXI,)), (23, (TAXI,)) ),
            ( (4, (TAXI,)), (14, (TAXI, BUS)), (24, (TAXI,)), (23, (TAXI, BUS)), (52, (BUS,)), 
                (89, (UNDERGROUND,)), (67, (UNDERGROUND,)), (46, (UNDERGROUND,)) ),
            ( (13, (TAXI, BUS)), (15, (TAXI, BUS)), (25, (TAXI,)) ),
            ( (5, (TAXI,)), (16, (TAXI,)), (28, (TAXI,)), (26, (TAXI,)), (14, (TAXI, BUS)),
                (29, (BUS,)), (41, (BUS,)) ),                                            # 015
            ( (5, (TAXI,)), (29, (TAXI,)), (28, (TAXI,)), (15, (TAXI,)) ),
            ( (7, (TAXI,)), (30, (TAXI,)), (29, (TAXI,)) ),
            ( (8, (TAXI,)), (31, (TAXI,)), (43, (TAXI,)) ),
            ( (8, (TAXI,)), (9, (TAXI,)), (32, (TAXI,)) ),
            ( (2, (TAXI,)), (9, (TAXI,)), (33, (TAXI,)) ),                              # 020
            ( (10, (TAXI,)), (33, (TAXI,)) ),
            ( (11, (TAXI,)), (23, (TAXI, BUS)), (35, (TAXI,)), (34, (TAXI, BUS)),
                (3, (BUS,)), (65, (BUS,)) ),
            ( (12, (TAXI,)), (13, (TAXI, BUS)), (37, (TAXI,)), (22, (TAXI, BUS)), 
                (3, (BUS,)), (67, (BUS,)) ),
            ( (13, (TAXI,)), (38, (TAXI,)), (37, (TAXI,)) ),
            ( (14, (TAXI,)), (39, (TAXI,)), (38, (TAXI,)) ),                            # 025
            ( (15, (TAXI,)), (27, (TAXI,)), (39, (TAXI,)) ),
            ( (26, (TAXI,)), (28, (TAXI,)), (40, (TAXI,)) ),
            ( (15, (TAXI,)), (16, (TAXI,)), (41, (TAXI,)), (27, (TAXI,)) ),
            ( (6, (TAXI,)), (17, (TAXI,)), (42, (TAXI, BUS)), (41, (TAXI, BUS)), (16, (TAXI,)),
                (55, (BUS,)), (15, (BUS,)) ),
            ( (17, (TAXI,)), (42, (TAXI,)) ),                                           # 030
            ( (18, (TAXI,)), (44, (TAXI,)), (43, (TAXI,)) ),
            ( (19, (TAXI,)), (33, (TAXI,)), (45, (TAXI,)), (44, (TAXI,)) ),
            ( (20, (TAXI,)), (21, (TAXI,)), (46, (TAXI,)), (32, (TAXI,)) ),
            ( (10, (TAXI,)), (22, (TAXI, BUS)), (48, (TAXI,)), (47, (TAXI,)), (63, (BUS,)),
                (46, (BUS,)) ),
            ( (22, (TAXI,)), (36, (TAXI,)), (65, (TAXI,)), (48, (TAXI,)) ),             # 035
            ( (37, (TAXI,)), (49, (TAXI,)), (35, (TAXI,)) ),
            ( (23, (TAXI,)), (24, (TAXI,)), (50, (TAXI,)), (36, (TAXI,)) ),
            ( (24, (TAXI,)), (25, (TAXI,)), (51, (TAXI,)), (50, (TAXI,)) ),
            ( (26, (TAXI,)), (52, (TAXI,)), (51, (TAXI,)), (25, (TAXI,)) ),
            ( (27, (TAXI,)), (41, (TAXI,)), (53, (TAXI,)), (52, (TAXI,)) ),             # 040
            ( (28, (TAXI,)), (29, (TAXI, BUS)), (54, (TAXI,)), (40, (TAXI,)),
                (15, (BUS,)), (87, (BUS,)), (52, (BUS,)) ),
            ( (30, (TAXI,)), (56, (TAXI,)), (72, (TAXI, BUS)), (29, (TAXI, BUS)),
                (7, (BUS,)) ),
            ( (18, (TAXI,)), (31, (TAXI,)), (57, (TAXI,)) ),
            ( (32, (TAXI,)), (58, (TAXI,)), (31, (TAXI,)) ),
            ( (32, (TAXI,)), (46, (TAXI,)), (60, (TAXI,)), (59, (TAXI,)),               # 045
                (58, (TAXI,)) ),
            ( (33, (TAXI,)), (47, (TAXI,)), (61, (TAXI,)), (45, (TAXI,)), (34, (BUS,)),
                (78, (BUS,)), (58, (BUS,)), (1, (BUS, UNDERGROUND)), (13, (UNDERGROUND,)),
                (79, (UNDERGROUND,)), (74, (UNDERGROUND,)) ),
            ( (34, (TAXI,)), (62, (TAXI,)), (46, (TAXI,)) ),
            ( (34, (TAXI,)), (35, (TAXI,)), (63, (TAXI,)), (62, (TAXI,)) ),
            ( (36, (TAXI,)), (50, (TAXI,)), (66, (TAXI,)) ),
            ( (37, (TAXI,)), (38, (TAXI,)), (49, (TAXI,)) ),                            # 050
            ( (38, (TAXI,)), (39, (TAXI,)), (52, (TAXI,)), (68, (TAXI,)), (67, (TAXI,)) ),
            ( (39, (TAXI,)), (40, (TAXI,)), (69, (TAXI,)), (51, (TAXI,)), (13, (BUS,)),
                (41, (BUS,)), (86, (BUS,)), (67, (BUS,)) ),
            ( (40, (TAXI,)), (54, (TAXI,)), (69, (TAXI,)) ),
            ( (41, (TAXI,)), (55, (TAXI,)), (70, (TAXI,)), (53, (TAXI,)) ),
            ( (71, (TAXI,)), (54, (TAXI,)), (29, (BUS,)), (89, (BUS,)) ),               # 055
            ( (42, (TAXI,)), (91, (TAXI,)) ),
            ( (43, (TAXI,)), (58, (TAXI,)), (73, (TAXI,)) ),
            ( (45, (TAXI,)), (59, (TAXI,)), (75, (TAXI,)), (74, (TAXI, BUS)), (57, (TAXI,)),
                (44, (TAXI,)), (46, (BUS,)), (77, (BUS,)), (1, (BUS,)) ),
            ( (45, (TAXI,)), (76, (TAXI,)), (75, (TAXI,)), (58, (TAXI,)) ),
            ( (45, (TAXI,)), (61, (TAXI,)), (76, (TAXI,)) ),                            # 060
            ( (46, (TAXI,)), (62, (TAXI,)), (78, (TAXI,)), (76, (TAXI,)), (60, (TAXI,)) ),
            ( (47, (TAXI,)), (48, (TAXI,)), (79, (TAXI,)), (61, (TAXI,)) ),
            ( (48, (TAXI,)), (64, (TAXI,)), (80, (TAXI,)), (79, (TAXI, BUS)),
                (34, (BUS,)), (65, (BUS,)), (100, (BUS,)) ),
            ( (65, (TAXI,)), (81, (TAXI,)), (63, (TAXI,)) ),
            ( (35, (TAXI,)), (66, (TAXI,)), (82, (TAXI, BUS)), (64, (TAXI,)),           # 065
                (22, (BUS,)), (67, (BUS,)), (63, (BUS,)) ),                                           
            ( (49, (TAXI,)), (67, (TAXI,)), (82, (TAXI,)), (65, (TAXI,)) ),
            ( (51, (TAXI,)), (68, (TAXI,)), (84, (TAXI,)), (66, (TAXI,)), (23, (BUS,)),
                (52, (BUS,)), (102, (BUS,)), (82, (BUS,)), (65, (BUS,)),
                (13, (UNDERGROUND,)), (89, (UNDERGROUND,)), (111, (UNDERGROUND,)),
                (79, (UNDERGROUND,)) ),
            ( (51, (TAXI,)), (69, (TAXI,)), (85, (TAXI,)), (67, (TAXI,)) ),
            ( (52, (TAXI,)), (53, (TAXI,)), (86, (TAXI,)), (68, (TAXI,)) ),
            ( (54, (TAXI,)), (71, (TAXI,)), (87, (TAXI,)) ),                            # 070
            ( (55, (TAXI,)), (72, (TAXI,)), (89, (TAXI,)), (70, (TAXI,)) ),
            ( (42, (TAXI, BUS)), (91, (TAXI,)), (90, (TAXI,)), (71, (TAXI,)),
                (107, (BUS,)), (105, (BUS,)) ),
            ( (57, (TAXI,)), (74, (TAXI,)), (92, (TAXI,)) ),
            ( (58, (TAXI, BUS)), (75, (TAXI,)), (92, (TAXI,)), (73, (TAXI,)),
                (94, (BUS,)), (46, (UNDERGROUND,)) ),
            ( (58, (TAXI,)), (59, (TAXI,)), (94, (TAXI,)), (74, (TAXI,)) ),             # 075
            ( (59, (TAXI,)), (60, (TAXI,)), (61, (TAXI,)), (77, (TAXI,)) ),
            ( (78, (TAXI, BUS)), (96, (TAXI,)), (95, (TAXI,)), (76, (TAXI,)),
                (124, (BUS,)), (94, (BUS,)), (58, (BUS,)) ),
            ( (61, (TAXI,)), (79, (TAXI, BUS)), (97, (TAXI,)), (77, (TAXI, BUS)),
                (46, (BUS,)) ),
            ( (62, (TAXI,)), (63, (TAXI, BUS)), (98, (TAXI,)), (78, (TAXI, BUS)),
                (46, (UNDERGROUND,)), (67, (UNDERGROUND,)), (111, (UNDERGROUND,)),
                (93, (UNDERGROUND,)) ),
            ( (63, (TAXI,)), (100, (TAXI,)), (99, (TAXI,)) ),                           # 080
            ( (64, (TAXI,)), (82, (TAXI,)), (100, (TAXI,)) ),
            ( (65, (TAXI, BUS)), (66, (TAXI,)), (67, (BUS,)), (101, (TAXI,)), (140, (BUS,)),
                (81, (TAXI,)), (100, (BUS,)) ),
            ( (102, (TAXI,)), (101, (TAXI,)) ),
            ( (67, (TAXI,)), (85, (TAXI,)) ),
            ( (68, (TAXI,)), (103, (TAXI,)), (84, (TAXI,)) ),                           # 085
            ( (69, (TAXI,)), (52, (BUS,)), (87, (BUS,)), (104, (TAXI,)), (116, (BUS,)),
                (103, (TAXI,)), (102, (BUS,)) ),
            ( (70, (TAXI,)), (41, (BUS,)), (88, (TAXI,)), (105, (BUS,)), (86, (BUS,)) ),
            ( (89, (TAXI,)), (117, (TAXI,)), (87, (TAXI,)) ),
            ( (71, (TAXI,)), (55, (BUS,)), (13, (UNDERGROUND,)), (105, (TAXI, BUS)),
                (128, (UNDERGROUND,)), (88, (TAXI,)), (140, (UNDERGROUND,)),
                (67, (UNDERGROUND,)) ),
            ( (72, (TAXI,)), (91, (TAXI,)), (105, (TAXI,)) ),                           # 090 
            ( (56, (TAXI,)), (107, (TAXI,)), (105, (TAXI,)), (90, (TAXI,)), (72, (TAXI,)) ),
            ( (73, (TAXI,)), (74, (TAXI,)), (93, (TAXI,)) ),
            ( (92, (TAXI,)), (94, (TAXI, BUS)), (79, (UNDERGROUND,)) ),
            ( (74, (BUS,)), (75, (TAXI,)), (95, (TAXI,)), (77, (BUS,)), (93, (TAXI, BUS)) ),
            ( (77, (TAXI,)), (122, (TAXI,)), (94, (TAXI,)) ),                           # 095
            ( (77, (TAXI,)), (97, (TAXI,)), (109, (TAXI,)) ),
            ( (78, (TAXI,)), (98, (TAXI,)), (109, (TAXI,)), (96, (TAXI,)) ),
            ( (79, (TAXI,)), (99, (TAXI,)), (110, (TAXI,)), (97, (TAXI,)) ),
            ( (80, (TAXI,)), (112, (TAXI,)), (110, (TAXI,)), (98, (TAXI,)) ),
            ( (81, (TAXI,)), (82, (BUS,)), (101, (TAXI,)), (113, (TAXI,)),              # 100
                (112, (TAXI,)), (111, (BUS,)), (80, (TAXI,)), (63, (BUS,)) ),
            ( (83, (TAXI,)), (114, (TAXI,)), (100, (TAXI,)), (82, (TAXI,)) ),
            ( (67, (BUS,)), (103, (TAXI,)), (86, (BUS,)), (115, (TAXI,)), (127, (BUS,)),
                (83, (TAXI,)) ),
            ( (85, (TAXI,)), (86, (TAXI,)), (102, (TAXI,)) ),
            ( (86, (TAXI,)), (116, (TAXI,)) ),
            ( (90, (TAXI,)), (72, (BUS,)), (91, (TAXI,)), (106, (TAXI,)),               # 105
                (107, (BUS,)), (108, (TAXI, BUS)), (87, (BUS,)), (89, (TAXI, BUS)) ),
            ( (107, (TAXI,)), (105, (TAXI,)) ),
            ( (91, (TAXI,)), (72, (BUS,)), (119, (TAXI,)), (161, (BUS,)),
                (106, (TAXI,)), (105, (BUS,)) ),
            ( (105, (TAXI, BUS)), (119, (TAXI,)), (135, (BUS,)), (117, (TAXI,)),
                (116, (BUS,)), (115, (BLACK,)) ),
            ( (97, (TAXI,)), (110, (TAXI,)), (124, (TAXI,)), (96, (TAXI,)) ),
            ( (99, (TAXI,)), (111, (TAXI,)), (109, (TAXI,)), (98, (TAXI,)) ),           # 110
            ( (112, (TAXI,)), (100, (BUS,)), (67, (UNDERGROUND,)),
                (153, (UNDERGROUND,)), (124, (TAXI, BUS)), (163, (UNDERGROUND,)),
                (110, (TAXI,)), (79, (UNDERGROUND,)) ),
            ( (100, (TAXI,)), (125, (TAXI,)), (111, (TAXI,)), (99, (TAXI,)) ),
            ( (114, (TAXI,)), (125, (TAXI,)), (100, (TAXI,)) ),
            ( (101, (TAXI,)), (115, (TAXI,)), (126, (TAXI,)),
                (132, (TAXI,)), (131, (TAXI,)), (113, (TAXI,)) ),
            ( (102, (TAXI,)), (127, (TAXI,)), (126, (TAXI,)), (114, (TAXI,)),           # 115
                (108, (BLACK,)), (157, (BLACK,)) ),
            ( (104, (TAXI,)), (86, (BUS,)), (117, (TAXI,)), (108, (BUS,)),
                (118, (TAXI,)), (142, (BUS,)), (127, (TAXI, BUS)) ),
            ( (88, (TAXI,)), (108, (TAXI,)), (129, (TAXI,)), (116, (TAXI,)) ),
            ( (116, (TAXI,)), (129, (TAXI,)), (142, (TAXI,)), (134, (TAXI,)) ),
            ( (107, (TAXI,)), (136, (TAXI,)), (108, (TAXI,)) ),
            ( (121, (TAXI,)), (144, (TAXI,)) ),                                         # 120
            ( (122, (TAXI,)), (145, (TAXI,)), (120, (TAXI,)) ),
            ( (95, (TAXI,)), (123, (TAXI, BUS)), (146, (TAXI,)),
                (121, (TAXI,)), (144, (BUS,)) ),
            ( (124, (TAXI, BUS)), (149, (TAXI,)), (165, (BUS,)), (148, (TAXI,)),
                (137, (TAXI,)), (144, (BUS,)), (122, (TAXI, BUS)) ),
            ( (109, (TAXI,)), (111, (TAXI, BUS)), (130, (TAXI,)), (138, (TAXI,)),
                (153, (BUS,)), (123, (TAXI, BUS)), (77, (BUS,)) ),
            ( (113, (TAXI,)), (131, (TAXI,)), (112, (TAXI,)) ),                         # 125
            ( (115, (TAXI,)), (127, (TAXI,)), (140, (TAXI,)), (114, (TAXI,)) ),
            ( (116, (TAXI, BUS)), (134, (TAXI,)), (133, (TAXI, BUS)),
                (126, (TAXI,)), (115, (TAXI,)), (102, (BUS,)) ),
            ( (143, (TAXI,)), (135, (BUS,)), (89, (UNDERGROUND,)), (160, (TAXI,)),
                (161, (BUS,)), (188, (TAXI,)), (199, (BUS,)), (172, (TAXI,)),
                (187, (BUS,)), (185, (UNDERGROUND,)), (142, (TAXI, BUS)),
                (140, (UNDERGROUND,)) ),
            ( (117, (TAXI,)), (135, (TAXI,)), (143, (TAXI,)), (142, (TAXI,)),
                (118, (TAXI,)) ),
            ( (131, (TAXI,)), (139, (TAXI,)), (124, (TAXI,)) ),                         # 130
            ( (114, (TAXI,)), (130, (TAXI,)), (125, (TAXI,)) ),
            ( (114, (TAXI,)), (140, (TAXI,)) ),
            ( (127, (TAXI, BUS)), (141, (TAXI,)), (157, (BUS,)), (140, (TAXI, BUS)) ),
            ( (118, (TAXI,)), (142, (TAXI,)), (141, (TAXI,)), (127, (TAXI,)) ),
            ( (108, (BUS,)), (136, (TAXI,)), (161, (TAXI, BUS)), (143, (TAXI,)),        # 135
                (128, (BUS,)), (129, (TAXI,)) ),
            ( (119, (TAXI,)), (162, (TAXI,)), (135, (TAXI,)) ),
            ( (123, (TAXI,)), (147, (TAXI,)) ),
            ( (152, (TAXI,)), (150, (TAXI,)), (124, (TAXI,)) ),
            ( (130, (TAXI,)), (140, (TAXI,)), (154, (TAXI,)), (153, (TAXI,)) ),
            ( (132, (TAXI,)), (82, (BUS,)), (126, (TAXI,)), (89, (UNDERGROUND,)),       # 140
                (133, (TAXI, BUS)), (128, (UNDERGROUND,)), (156, (TAXI, BUS)),
                (154, (TAXI, BUS)), (153, (UNDERGROUND,)), (139, (TAXI,)) ),
            ( (134, (TAXI,)), (142, (TAXI,)), (158, (TAXI,)), (133, (TAXI,)) ),
            ( (118, (TAXI,)), (116, (BUS,)), (129, (TAXI,)), (143, (TAXI,)),
                (128, (TAXI, BUS)), (158, (TAXI,)), (157, (BUS,)), (141, (TAXI,)),
                (134, (TAXI,)) ),
            ( (135, (TAXI,)), (160, (TAXI,)), (128, (TAXI,)), (142, (TAXI,)),
                (129, (TAXI,)) ),
            ( (120, (TAXI,)), (122, (BUS,)), (145, (TAXI,)), (123, (BUS,)),
                (163, (BUS,)), (177, (TAXI,)) ),
            ( (121, (TAXI,)), (146, (TAXI,)), (144, (TAXI,)) ),                         # 145
            ( (122, (TAXI,)), (147, (TAXI,)), (163, (TAXI,)), (145, (TAXI,)) ),
            ( (137, (TAXI,)), (164, (TAXI,)), (146, (TAXI,)) ),
            ( (123, (TAXI,)), (149, (TAXI,)), (164, (TAXI,)) ),
            ( (123, (TAXI,)), (150, (TAXI,)), (165, (TAXI,)), (148, (TAXI,)) ),
            ( (138, (TAXI,)), (151, (TAXI,)), (149, (TAXI,)) ),                         # 150
            ( (152, (TAXI,)), (166, (TAXI,)), (165, (TAXI,)), (150, (TAXI,)) ),
            ( (153, (TAXI,)), (151, (TAXI,)), (138, (TAXI,)) ),
            ( (139, (TAXI,)), (111, (UNDERGROUND,)), (154, (TAXI, BUS)),
                (140, (UNDERGROUND,)), (167, (TAXI,)), (184, (BUS,)),
                (185, (UNDERGROUND,)), (166, (TAXI,)), (180, (BUS,)),
                (163, (UNDERGROUND,)), (152, (TAXI,)), (124, (BUS,)) ),
            ( (140, (TAXI, BUS)), (155, (TAXI,)), (156, (BUS,)), (153, (TAXI, BUS)),
                (139, (TAXI,)) ),
            ( (156, (TAXI,)), (168, (TAXI,)), (167, (TAXI,)), (154, (TAXI,)) ),         # 155
            ( (140, (TAXI, BUS)), (157, (TAXI, BUS)), (169, (TAXI,)),
                (184, (BUS,)), (155, (TAXI,)), (154, (BUS,)) ),
            ( (133, (BUS,)), (158, (TAXI,)), (142, (BUS,)), (170, (TAXI,)),
                (185, (BUS,)), (156, (TAXI, BUS)), (115, (BLACK,)), (194, (BLACK,)) ),
            ( (141, (TAXI,)), (142, (TAXI,)), (159, (TAXI,)), (157, (TAXI,)) ),
            ( (158, (TAXI,)), (172, (TAXI,)), (198, (TAXI,)), (186, (TAXI,)),
                (170, (TAXI,)) ),
            ( (143, (TAXI,)), (161, (TAXI,)), (173, (TAXI,)), (128, (TAXI,)) ),         # 160
            ( (107, (BUS,)), (174, (TAXI,)), (199, (BUS,)), (160, (TAXI,)),
                (128, (BUS,)), (135, (TAXI, BUS)) ),
            ( (175, (TAXI,)), (136, (TAXI,)) ),
            ( (146, (TAXI,)), (111, (UNDERGROUND,)), (153, (UNDERGROUND,)),
                (191, (BUS,)), (177, (TAXI,)), (176, (BUS,)), (144, (BUS,)) ),
            ( (147, (TAXI,)), (148, (TAXI,)), (179, (TAXI,)), (178, (TAXI,)) ),
            ( (149, (TAXI,)), (123, (BUS,)), (151, (TAXI,)), (180, (TAXI, BUS)),        # 165
                (179, (TAXI,)), (191, (BUS,)) ),
            ( (153, (TAXI,)), (183, (TAXI,)), (181, (TAXI,)), (151, (TAXI,)) ),
            ( (155, (TAXI,)), (168, (TAXI,)), (183, (TAXI,)), (153, (TAXI,)) ),
            ( (155, (TAXI,)), (184, (TAXI,)), (167, (TAXI,)) ),
            ( (156, (TAXI,)), (184, (TAXI,)) ),
            ( (157, (TAXI,)), (159, (TAXI,)), (185, (TAXI,)) ),                         # 170
            ( (173, (TAXI,)), (175, (TAXI,)), (199, (TAXI,)) ),
            ( (128, (TAXI,)), (187, (TAXI,)), (159, (TAXI,)) ),
            ( (160, (TAXI,)), (174, (TAXI,)), (171, (TAXI,)), (188, (TAXI,)) ),
            ( (175, (TAXI,)), (173, (TAXI,)), (161, (TAXI,)) ),
            ( (162, (TAXI,)), (171, (TAXI,)), (174, (TAXI,)) ),                         # 175
            ( (177, (TAXI,)), (163, (BUS,)), (189, (TAXI,)), (190, (BUS,)) ),
            ( (144, (TAXI,)), (163, (TAXI,)), (176, (TAXI,)) ),
            ( (164, (TAXI,)), (191, (TAXI,)), (189, (TAXI,)) ),
            ( (165, (TAXI,)), (191, (TAXI,)), (164, (TAXI,)) ),
            ( (165, (TAXI, BUS)), (181, (TAXI,)), (153, (BUS,)), (193, (TAXI,)),        # 180
                (184, (BUS,)), (190, (BUS,)) ),
            ( (166, (TAXI,)), (182, (TAXI,)), (193, (TAXI,)), (180, (TAXI,)) ),
            ( (183, (TAXI,)), (195, (TAXI,)), (181, (TAXI,)) ), 
            ( (167, (TAXI,)), (196, (TAXI,)), (182, (TAXI,)), (166, (TAXI,)) ),
            ( (169, (TAXI,)), (156, (BUS,)), (185, (TAXI, BUS)), (197, (TAXI,)),
                (196, (TAXI,)), (180, (BUS,)), (168, (TAXI,)), (153, (BUS,)) ),
            ( (170, (TAXI,)), (157, (BUS,)), (186, (TAXI,)), (187, (BUS,)),             # 185
                (128, (UNDERGROUND,)), (184, (TAXI, BUS)), (153, (UNDERGROUND,)) ),
            ( (159, (TAXI,)), (198, (TAXI,)), (185, (TAXI,)) ),
            ( (172, (TAXI,)), (128, (BUS,)), (188, (TAXI,)), (198, (TAXI,)),
                (185, (BUS,)) ),
            ( (128, (TAXI,)), (173, (TAXI,)), (199, (TAXI,)), (187, (TAXI,)) ),
            ( (178, (TAXI,)), (190, (TAXI,)), (176, (TAXI,)) ),
            ( (191, (TAXI, BUS)), (192, (TAXI,)), (180, (BUS,)),                        # 190
                (189, (TAXI,)), (176, (BUS,)) ),
            ( (179, (TAXI,)), (165, (BUS,)), (192, (TAXI,)), (190, (TAXI, BUS)),
                (178, (TAXI,)), (163, (BUS,)) ),
            ( (191, (TAXI,)), (194, (TAXI,)), (190, (TAXI,)) ),
            ( (181, (TAXI,)), (194, (TAXI,)), (180, (TAXI,)) ),
            ( (195, (TAXI,)), (192, (TAXI,)), (193, (TAXI,)), (157, (BLACK,)) ),
            ( (182, (TAXI,)), (197, (TAXI,)), (194, (TAXI,)) ),                         # 195
            ( (183, (TAXI,)), (184, (TAXI,)), (197, (TAXI,)) ),
            ( (196, (TAXI,)), (184, (TAXI,)), (195, (TAXI,)) ),
            ( (159, (TAXI,)), (187, (TAXI,)), (199, (TAXI,)), (186, (TAXI,)) ),
            ( (188, (TAXI,)), (128, (BUS,)), (171, (TAXI,)), (161, (BUS,)),
                (198, (TAXI,)) ) 
        )
    #state = [140, 5, [34, 29, 117, 174, 112], [[10, 8, 4], [10, 8, 4], [10, 8, 4], [10, 8, 4], [10, 8, 4]], 1]
    feature_vector = generate_feature_space(state)
    print(feature_vector)
    feature_vector_tensor = torch.tensor(feature_vector)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN().to(device)
    policy_net = torch.load("/Users/shreyasi/Desktop/599-FINAL/londonlaw/londonlaw/aiclients/mr_x_model.pth")
    #result = (policy_net(feature_vector_tensor))
    returned_states = policy_net(feature_vector_tensor)
    max_value = float("-inf")
    max_index = 0

    # print("POSSIBLE MOVES: ", possible_moves)
    possible_moves_indexes = []
    possible_moves = board[state[0]]
    for i in possible_moves:
        possible_moves_indexes.append(i[0])

    for index, value in enumerate(returned_states):
        if(index in possible_moves_indexes and value > max_value):
            max_value = value
            max_index = index
            
    # max_index_tensor = torch.tensor(max_index)
    best_transport = "TAXI"
    for i in possible_moves:
        if(i[0] == max_index):
            best_transport = i[1][0]
    
    print(max_index)
    print(str(best_transport))

    return max_index,best_transport
#print ('Result',max_index_tensor)
# return max_index_tensor
#return policy_net(torch.tensor(state)).argmax(dim=1).to(self.device) # exploit
# print(moves)

#predict_best_move([140, 5, [34, 29, 117, 174, 112], [[10, 8, 4], [10, 8, 4], [10, 8, 4], [10, 8, 4], [10, 8, 4]], 1])



