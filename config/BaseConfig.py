class BaseConfig():
    ModelName = "CharTCN_NonLocal"
    TrainPath = "/home/longred/TextClassification/dataset/preProcess/E_commerce_data/train_data.tsv"
    ValPath = "/home/longred/TextClassification/dataset/preProcess/E_commerce_data/val_data.tsv"
    DatasetName = "ECommerce"
    CharVectorsPath = "/home/longred/TextClassification/dataset/embedding/char2Vec_300.txt"
    WordVectorsPath = "/home/longred/TextClassification/dataset/embedding/word2Vec_300.txt"
    CharVectors = None
    WordVectors = None
    CharVectorsDim = 300
    WordVectorsDim = 300
    CharVocabSize = None
    WordVocabSize = None
    LogPath = ""
    Label = {"last": 1258, "middle": 192, "first": 22}
    SamplesPerCls = {
        "last":[27, 215, 12, 30, 32, 24, 34, 43, 344, 54, 42, 73, 32, 146, 18, 49, 134, 11, 107, 73, 86, 24, 50, 59, 34, 17, 22, 349, 127, 326, 684, 45, 940, 249, 1074, 720, 381, 309, 246, 17, 1512, 318, 1207, 1648, 162, 524, 88, 41, 24, 10, 80, 154, 17, 24, 52, 20, 94, 32, 90, 211, 1402, 147, 9, 680, 1266, 11, 159, 339, 29, 879, 36, 126, 146, 21, 173, 11, 274, 81, 9, 115, 2237, 14, 380, 10, 10, 87, 7, 62, 101, 209, 98, 355, 15, 171, 13, 950, 20, 24, 122, 218, 24, 29, 421, 164, 780, 211, 26, 9, 96, 94, 297, 486, 102, 92, 225, 9, 67, 125, 158, 15, 88, 315, 179, 406, 10, 201, 7, 8, 25, 63, 118, 295, 577, 687, 255, 89, 18783, 114, 141, 91, 109, 1601, 406, 40, 130, 884, 74, 14, 194, 245, 248, 51, 13, 664, 8, 166, 16, 33, 51, 111, 22, 77, 14, 275, 153, 954, 59, 366, 318, 40, 5146, 54, 1758, 7095, 12, 3196, 1973, 28, 5015, 8, 85, 94, 66, 74, 236, 10, 20, 11, 509, 34, 83, 42, 11, 79, 48, 356, 56, 2038, 12069, 1006, 1028, 1021, 2258, 2278, 7, 14, 17, 669, 174, 45, 7, 94, 31, 16, 38, 16, 19, 16, 9, 24, 8, 122, 31, 63, 64, 11, 39, 46, 38, 104, 20, 18, 32, 13, 28, 59, 26, 9, 30, 9, 27, 78, 22, 10, 18, 181, 7, 13, 40, 111, 26, 41, 36, 24, 125, 1032, 39, 9, 36, 90, 160, 3228, 141, 99, 895, 21, 133, 21, 58, 8, 56, 48, 34, 111, 204, 35, 317, 22, 16, 65, 22, 17, 9, 17, 103, 8, 35, 13, 14, 32, 51, 360, 38, 44, 50, 15, 20, 11, 10, 8, 8, 13, 231, 6, 170, 196, 31, 110, 1039, 1443, 115, 143, 419, 221, 113, 26, 23, 14, 45, 643, 112, 9, 61, 191, 359, 24, 405, 21, 13, 76, 1714, 70, 4185, 1230, 103, 10, 584, 17, 8, 411, 57, 218, 810, 281, 557, 507, 1933, 209, 9, 11, 8, 108, 129, 24, 62, 41, 185, 417, 79, 17, 120, 22, 64, 32, 23, 58, 270, 290, 201, 132, 17, 75, 110, 620, 64, 108, 73, 68, 128, 17, 95, 127, 48, 39, 134, 23, 150, 35, 359, 29, 232, 7, 84, 1726, 30, 13, 70, 7, 12, 14, 15, 49, 6, 77, 773, 27, 173, 84, 21, 93, 16, 23, 15, 10, 173, 56, 41, 854, 76, 20, 1427, 80, 9, 22, 87, 34, 24, 108, 71, 10258, 2054, 961, 9374, 501, 1386, 7816, 80, 463, 524, 9414, 1948, 14, 25, 1854, 91, 12, 38, 458, 88, 15, 2008, 227, 64, 651, 867, 87, 64, 132, 180, 1737, 13, 17, 12, 33, 43, 43, 44, 13, 10, 20, 48, 8, 10, 45, 15, 156, 143, 101, 16, 97, 167, 462, 297, 66, 200, 111, 103, 233, 823, 369, 363, 447, 43, 573, 667, 22, 174, 62, 223, 41, 131, 65, 623, 45, 111, 661, 234, 8, 21, 112, 117, 7, 97, 140, 63, 64, 47, 23, 17, 49, 11, 20, 76, 66, 17, 15, 75, 21, 73, 166, 94, 71, 31, 17375, 10, 190, 24, 1994, 10, 161, 386, 120, 62, 171, 51, 73, 259, 121, 18, 220, 15, 72, 533, 167, 837, 92, 51, 141, 201, 41, 29, 9, 10, 18, 43, 23, 10, 18, 15, 8, 8, 34, 653, 53, 8, 28, 12, 8, 9, 60, 17, 10, 29, 13, 6, 21, 88, 9, 11, 52, 10, 30, 18, 33, 10, 10, 237, 116, 10, 9, 26, 41, 39, 20, 8, 23, 27, 20, 40, 7, 23, 9, 24, 66, 129, 24, 74, 24, 10, 106, 15, 8, 107, 20, 132, 51, 18, 19, 11, 92, 22, 30, 10, 10, 9, 38, 40, 21, 10, 34, 185, 25, 39, 27, 13, 10, 27, 30, 85, 27, 10, 40, 10, 27, 17, 8, 30, 44, 18, 94, 482, 12, 15, 11, 47, 13, 15, 11, 169, 13, 12, 10, 169, 57, 35, 15, 94, 15, 19, 10, 55, 20, 88, 15, 249, 15, 13, 85, 9, 23, 41, 10, 87, 15, 9, 23, 19, 8, 53, 13, 12, 25, 20, 48, 17, 329, 115, 68, 8, 140, 49, 201, 24, 43, 29, 24, 48, 73, 111, 48, 183, 36, 20, 56, 8, 9, 8, 15, 8, 7, 7, 14, 35, 20, 13, 8, 80, 17, 14, 32, 17, 9, 49, 22, 48, 8, 31, 22, 31, 15, 27, 15, 23, 21, 15, 10, 12, 14, 94, 36, 194, 318, 40, 52, 8, 17, 22, 24, 22, 12, 24, 9, 14, 7, 21, 10, 52, 12, 17, 22, 20, 63, 17, 12, 129, 34, 62, 11, 55, 72, 9, 43, 25, 35, 28, 15, 22, 57, 25, 8, 10, 20, 145, 1400, 481, 446, 45, 55, 46, 91, 153, 13, 281, 93, 11, 21, 8, 9, 32, 10, 10, 19, 36, 12, 45, 11, 18, 29, 7, 90, 22, 32, 15, 11, 39, 11, 8, 12, 63, 8, 12, 33, 29, 9, 9, 10, 13, 269, 120, 44, 87, 17, 119, 9, 27, 27, 14, 33, 10, 284, 125, 27, 34, 37, 10, 146, 73, 17, 9, 9, 33, 31, 28, 48, 45, 127, 66, 62, 15, 108, 85, 68, 36, 13, 12, 24, 37, 10, 8, 11, 11, 17, 10, 60, 34, 68, 33, 55, 23, 8, 7, 157, 8, 27, 129, 22, 36, 87, 103, 11, 10, 27, 8, 23, 183, 18, 40, 13, 63, 11, 38, 11, 11, 10, 31, 16, 13, 12, 21, 29, 8, 7, 11, 42, 7, 87, 12, 45, 18, 8, 8, 8, 7, 16, 21, 22, 129, 12, 183, 7, 12, 26, 13, 14, 8, 13, 17, 7, 51, 27, 184, 148, 92, 157, 28, 440, 43, 365, 45, 71, 91, 87, 28, 94, 149, 90, 73, 8, 123, 31, 264, 108, 67, 22, 12, 13, 6326, 24, 636, 50, 134, 2787, 47, 227, 1412, 76, 21, 1128, 19, 57, 281, 841, 24, 1683, 449, 302, 70, 7, 65, 77, 9, 145, 85, 22, 127, 94, 37, 45, 36, 15, 11, 9, 24, 10, 12, 11, 26, 29, 10, 823, 13, 19, 40, 173, 26, 17, 7, 41, 34, 8, 31, 57, 15, 71, 27, 40, 60, 8, 17, 38, 23, 59, 45, 16, 8, 59, 475, 12, 15, 113, 176, 51, 6, 8, 50, 75, 93, 27, 10, 15, 10, 41, 169, 118, 16, 101, 110, 6, 9, 32, 12, 10, 13, 22, 33, 10, 8, 37, 45, 9, 90, 10, 13, 13, 13, 12, 11, 31, 8, 23, 37, 21, 20, 7, 10, 13, 31, 37, 848, 204, 1989, 305, 19, 551, 18, 59, 484, 12, 414, 76, 255, 29, 108, 41, 43, 320, 1345, 2948, 401, 342, 309, 682, 208, 854, 85, 13, 51, 129, 461, 713, 821, 108, 163, 39, 14, 374, 308, 66, 809, 63, 29, 36, 43, 724, 185, 941, 13, 329, 31, 345, 123, 114, 1062, 87, 246, 1154, 440, 220, 62, 71, 8, 148, 8, 136, 42, 16, 106, 839, 74, 30, 45, 4304, 76, 10181, 45, 440, 1454, 63, 92, 4544, 1816, 300, 13, 1540, 12, 619, 23, 17, 373, 21, 899, 101, 16, 2491, 15, 258, 365, 334, 290, 269, 41, 18, 25, 11, 14, 10, 14, 119, 107, 50, 25, 20, 13, 22, 9, 38, 66, 30, 232, 66, 7, 153, 87, 1022, 1287, 1462, 7, 700, 854, 1590, 184, 44, 10, 1585, 909, 64, 145, 31, 98, 29, 13, 37, 456, 17, 120, 18, 375, 3709],
        "middle":[242, 42, 133, 513, 196, 301, 159, 206, 1531, 5448, 3859, 243, 694, 4042, 1871, 2807, 101, 1831, 417, 2778, 1813, 21238, 4896, 2400, 24325, 1814, 16141, 5557, 7, 31, 1020, 16, 98, 154, 292, 339, 365, 278, 4860, 1169, 170, 701, 125, 806, 49, 8, 250, 170, 227, 1149, 1701, 779, 846, 2864, 4255, 1343, 584, 493, 4515, 28, 1045, 1097, 224, 1043, 611, 805, 1817, 132, 1218, 1378, 1811, 71, 44779, 14, 8598, 161, 44, 13, 30, 66, 60, 680, 1136, 336, 3285, 718, 1674, 676, 100, 290, 20585, 120, 990, 1701, 434, 225, 754, 8, 165, 271, 646, 489, 472, 158, 10, 512, 50, 238, 1071, 228, 650, 230, 1566, 111, 97, 422, 752, 251, 95, 151, 236, 55, 306, 25, 8, 30, 3249, 518, 186, 665, 847, 787, 96, 510, 666, 71, 175, 310, 249, 351, 2813, 47, 10004, 3221, 3376, 742, 92, 888, 288, 264, 341, 561, 340, 232, 62, 344, 211, 15, 54, 78, 8, 37, 180, 25, 42, 8, 23, 139, 37, 848, 4927, 9362, 1944, 108, 3954, 2122, 413, 1110, 24880, 2053, 4038, 41, 211, 202, 44, 679, 7106, 2548, 338, 79, 473, 4222],
        "first":[242, 1550, 15817, 9805, 56486, 21698, 1074, 1248, 7178, 1238, 25728, 57869, 314, 8565, 24220, 21822, 16601, 2615, 2396, 24788, 30971, 15943]
    }
    UseLabel = "last"
    UseInput = "char"
    learning_rate = 0.001
    TrainBatchSize = 64
    ValBatchSize = 512
    DeviceIds = [1]
    resume = None
    start_iter = 0
    TrainIterAll = 70000
    ValInter = 1000
if __name__ == "__main__":
    b = BaseConfig()
    print(b)
