import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
import unet_model as Net
import os
from skimage import measure
from argparse import ArgumentParser


torch.backends.cudnn.enabled = True
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#调色板，见https://www.rapidtables.org/zh-CN/web/color/RGB_Color.html
pallete = [[176,196,222],
           [230,230,250],
           [240,248,255],
           [240,255,240],
           [255,255,255],

           [255, 140, 0],
           [255,215,0],
           [238,232,170],
           [189,183,107],
           [255,255,0],

           [154,205,50],
           [124,252,0],
           [0,255,0],
           [50,205,50],
           [0,250,154],

           [0,255,255],
           [0,206,209],
           [64,224,208],
           [127,255,212],
           [176,224,230],

           [95,158,160],
           [70,130,180],
           [100,149,237],
           [0,191,255],
           [135,206,235],

           [216,191,216],
           [221,160,221],
           [238,130,238],
           [255,0,255],
           [219,112,147],

           [255,105,180],
           [255,182,193],
           [250,235,215],
           [245,245,220],
           [255,250,205],

           [210,105,30],
           [244,164,96],
           [210,180,140],
           [188,143,143],
           [255,228,225],

           [220, 20, 60],
           [255, 0, 0],
           [255, 99, 71],
           [255, 127, 80],
           [233, 150, 122]]


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img

def remove_small_points(img):
    # image:二值图像
    # threshold_point:符合面积条件大小的阈值
    img_label, num = measure.label(img, connectivity=1, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)
    tmpmax = props[0].area
    for i in range(1, len(props)): #找最大连通域的面积
        if props[i].area > tmpmax:
            tmpmax = props[i].area

    tmp_threshold = tmpmax/8
    for i in range(0, len(props)): #滤除小于最大连通域面积的一半的区域
        if props[i].area > tmp_threshold:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 1 #二值具体灰度值
    img_label, num = measure.label(resMatrix, connectivity=1, return_num=True)  # 输出二值图像中所有的连通域
    return resMatrix, num

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[stats[:,4].argsort()]
    return stats[:-1] # 排除最外层的连通图


def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1[0],box1[1],box1[2],box1[3]
    x11, y11, x12, y12 = box2[0],box2[1],box2[2],box2[3]

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def clear_Rectangle(args, image_list_result):
    kernel = np.ones((5, 5), np.uint8)
    filepath = args.savedir + os.sep + 'clear_data'
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    z = 0
    nummax = 1
    for i, imgName in enumerate(image_list_result):
        img0 = cv2.imread(imgName,0)
        #h00, w00 = img0.shape[:2]
        img = cv2.erode(img0,kernel,iterations = 3) #腐蚀
        img, num = remove_small_points(img)

        if num > nummax:
            nummax = num
            z = i

        name = imgName.split('\\')[-1]
        cv2.imwrite(args.savedir + os.sep + 'clear_data'+ os.sep + name.replace(args.img_extn, 'png'), img)
    return nummax, z

    #返回鼠脑数目，和拥有完整各鼠脑的图片号


def search_Rectangle(args, image_list_clear, nummax, z):
    kernel = np.ones((5, 5), np.uint8)
    image_list_clear1 = image_list_clear[:z]
    image_list_clear2 = image_list_clear[z+1:]

    cutp = [None] * (nummax*4) #创建保存初始剪切位置的列表
    s0 = [None] * nummax
    imgName = image_list_clear[z]
    img0 = cv2.imread(imgName, 0)
    bboxs = mask_find_bboxs(img0)
    k = 0
    for b in bboxs:
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
        cutp[0+4*k], cutp[1+4*k], cutp[2+4*k], cutp[3+4*k] = x0, y0, x1, y1
        s0[k] = (b[2]+1)*(b[3]+1)
        k = k+1
    #以上为初始化剪切位置和初始化面积
    for i, imgName in enumerate(image_list_clear1):
        img1 = cv2.imread(imgName, 0)
        bboxs1 = mask_find_bboxs(img1)
        for b in bboxs1:
            x0, y0 = b[0], b[1]
            x1 = b[0] + b[2]
            y1 = b[1] + b[3]

            box2 = [None] * 4
            box2[0],box2[1],box2[2],box2[3]=x0, y0, x1, y1
            box1 = [None] * 4
            j=0
            for k in range(0, nummax):
                #计算交集面积
                box1[0], box1[1], box1[2], box1[3] = cutp[0+4*k], cutp[1+4*k], cutp[2+4*k], cutp[3+4*k]
                try:
                    if mat_inter(box1, box2) == True:
                        #更新cutp
                        if box2[0] < cutp[0+4*k]:
                            cutp[0+4*k] = box2[0]
                        if box2[1] < cutp[1+4*k]:
                            cutp[1+4*k] = box2[1]
                        if box2[2] > cutp[2+4*k]:
                            cutp[2+4*k] = box2[2]
                        if box2[3] > cutp[3+4*k]:
                            cutp[3+4*k] = box2[3]
                        j = 1
                except:
                    j = 1
            if j == 0:
                nummax = nummax +1
                cutp.extend([box2[0],box2[1],box2[2],box2[3]])

    for i, imgName in enumerate(image_list_clear2):
        img1 = cv2.imread(imgName, 0)
        bboxs1 = mask_find_bboxs(img1)
        for b in bboxs1:
            x0, y0 = b[0], b[1]
            x1 = b[0] + b[2]
            y1 = b[1] + b[3]

            box2 = [None] * 4
            box2[0],box2[1],box2[2],box2[3]=x0, y0, x1, y1
            box1 = [None] * 4
            j = 0
            for k in range(0, nummax):
                #计算交集面积
                box1[0], box1[1], box1[2], box1[3] = cutp[0+4*k], cutp[1+4*k], cutp[2+4*k], cutp[3+4*k]
                try:
                    if mat_inter(box1, box2) == True:
                        # 更新cutp
                        if box2[0] < cutp[0 + 4 * k]:
                            cutp[0 + 4 * k] = box2[0]
                        if box2[1] < cutp[1 + 4 * k]:
                            cutp[1 + 4 * k] = box2[1]
                        if box2[2] > cutp[2 + 4 * k]:
                            cutp[2 + 4 * k] = box2[2]
                        if box2[3] > cutp[3 + 4 * k]:
                            cutp[3 + 4 * k] = box2[3]
                        j = 1
                except:
                    j = 1
            if j == 0:
                nummax = nummax +1
                cutp.extend([box2[0],box2[1],box2[2],box2[3]])
    # 对重复的剪切参数框进行进一步筛选
    #异常值滤除，（例如：None）
    for k in range(0, nummax):
        if cutp[0+4*k] is None :
            del cutp[(0+4*k):(3+4*k)]
            nummax = nummax - 1
    #重复框滤除
    r = 0 #重复个数
    re = [None] * 1 #重复的鼠脑编号
    for k in range(0, nummax):
        for n in range(k+1, nummax):
            if cutp[0+4*k] == cutp[0+4*n] and cutp[1+4*k] == cutp[1+4*n] and cutp[2+4*k] == cutp[2+4*n]:
                r = r+1  #第r个重复
                if r == 1:
                    re[0] = n #重复的鼠脑编号
                if r != 1:
                    re.extend([n])
    if r != 0:
        #可能存在不止重复一次的鼠脑，处理
        res = []
        for i in re:
            if i not in res:
                res.append(i)
        #将鼠脑序号换成坐标
        rep = [None] * len(res) *4
        for k in range(0, len(res)):
            rep[0+4*k] = 0 + 4 * res[k]
            rep[1+4*k] = 1 + 4 * res[k]
            rep[2+4*k] = 2 + 4 * res[k]
            rep[3+4*k] = 3 + 4 * res[k]
        cutp = np.delete(cutp, rep).tolist()
        nummax = nummax -len(res)
    #以上，剪切参数框检测完成，将框进一步放大后进行绘制保存
    file_handle = open(args.savedir + os.sep+'CutParameter.txt',mode='a+')
    for k in range(0, nummax):
        cutp[0+4*k] = cutp[0+4*k] - 10
        cutp[1+4*k] = cutp[1+4*k] - 10
        cutp[2+4*k] = cutp[2+4*k] + 10
        cutp[3+4*k] = cutp[3+4*k] + 10
        file_handle.write(str(cutp[0 + 4 * k]) + '\n')
        file_handle.write(str(cutp[2 + 4 * k]) + '\n')
        file_handle.write(str(cutp[1 + 4 * k]) + '\n')
        file_handle.write(str(cutp[3 + 4 * k]) + '\n')

    #dilation = cv2.dilate(img, kernel, iterations=5)  # 对最大矩形框做膨胀
    file_handle.close()
    return cutp

def save_draw_Rectangle(args, cutp, image_list):
    filepath = args.savedir + os.sep + 'rectangle_data'
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    for i, imgName in enumerate(image_list):
        img = cv2.imread(imgName,0)
        mask_BGR = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为3通道图，使得color能够显示红色。
        #color = (0, 0, 255)  # Red color in BGR；红色：rgb(255,0,0)
        thickness = 1  # Line thickness of 1 px

        nummax = int(len(cutp) / 4)
        for k in range(0, nummax):
            color = pallete[k]
            start_point, end_point = (cutp[0+4*k], cutp[1+4*k]), (cutp[2+4*k], cutp[3+4*k])
            mask_BGR = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            s = str(k)

            mask_BGR = cv2.putText(mask_BGR, s, (cutp[0+4*k]+5, cutp[3+4*k]-5), font, 0.8, color, 2)
            # 添加文字，0.8表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，1表示粗细

        name = imgName.split('/')[-1]
        cv2.imwrite(args.savedir + os.sep + 'rectangle_' + name.replace(args.img_extn, 'png'), mask_BGR)

def evaluateModel(args, model, up, image_list):
    # gloabl mean and std values
    mean = [8.433324,8.433324,8.433324]
    std = [20.228735,20.228735,20.228735]
    
    filepath = args.savedir + os.sep + 'gray_data'
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    for i, imgName in enumerate(image_list):
        img = cv2.imread(imgName)
        h00, w00 = img.shape[:2]
        #img = np.rot90(img, 1)# n=0,1,2,3,... 即逆时针旋转0，90,180,270
        h0, w0 = img.shape[:2]
        if args.overlay:
            img_orig = np.copy(img)
            img_orig = np.reshape(img_orig, (h0, w0, 1))#转换为三维numpy模式

        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]

        # resize the image to
        img = cv2.resize(img, (512, 512))
        h1, w1 = img.shape[:2]
        img = np.reshape(img, (h1, w1, 3))  # 转换为三维numpy模式

        if args.overlay:
            img_orig = cv2.resize(img_orig, (512, 512))
            h1, w1 = img_orig.shape[:2]
            img_orig = np.reshape(img_orig, (h1, w1, 3))  # 转换为三维numpy模式

        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        img_variable = Variable(img_tensor, volatile=True)
        if args.gpu:
            img_variable = img_variable.cuda()
        img_out = model(img_variable)
        '''
        imgddgf1 = imgddgf[:, 0, :, :]
        print(imgddgf1.shape)
        imgddgf1 = imgddgf1.view(imgddgf1.shape[1], imgddgf1.shape[2])
        print(imgddgf1.shape)
        feature = imgddgf1.cpu().data.numpy()
        # use sigmod to [0,1]
        feature = 1.0 / (1 + np.exp(-1 * feature))
        # to [0,255]
        feature = np.round(feature * 255)
        feature = feature
        print(feature[0])
        '''
        #if args.modelType == 2:
        #    img_out = up(img_out)

        classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
        if i % 1 == 0:
            print(i)
        name = imgName.split('/')[-1]
        classMap_numpy = cv2.resize(classMap_numpy, (w0, h0))
        #classMap_numpy = np.rot90(classMap_numpy, -1)  # n=0,1,2,3,... 即顺时针旋转0，90,180,270
        yyyyy = args.savedir + os.sep + 'gray_' + name.replace(args.img_extn, 'png')
        cv2.imwrite(args.savedir + os.sep + 'gray_' + name.replace(args.img_extn, 'png'), classMap_numpy)
        #cv2.imwrite(args.savedir + os.sep + 'grayddgf_' + name.replace(args.img_extn, 'png'), feature)

        if args.colored:
            classMap_numpy_color = np.zeros((h00, w00, 3), dtype=np.uint8)
            for idx in range(len(pallete)):
                [r, g, b] = pallete[idx]
                classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
            savename = args.savedir + os.sep + 'c_' + name.replace(args.img_extn, 'png')
            cv2.imwrite(savename, classMap_numpy_color)
            if args.overlay:
                overlayed = cv2.addWeighted(img_orig, 0.5, classMap_numpy_color, 0.5, 0) #原图像为单通道，和彩色标签图无法融合
                cv2.imwrite(args.savedir + os.sep + 'over_' + name.replace(args.img_extn, 'jpg'), overlayed)

        if args.cityFormat:
            classMap_numpy = relabel(classMap_numpy.astype(np.uint8))
        classMap_numpy = cv2.resize(classMap_numpy, (w0, h0))
        #classMap_numpy = np.rot90(classMap_numpy, -1)  # n=0,1,2,3,... 即顺时针旋转0，90,180,270
        print(args.savedir + os.sep + name.replace(args.img_extn, 'png'))
        #cv2.imwrite(args.savedir + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)


def main(args):
    # read all the images in the folder
    image_list = glob.glob(args.data_dir + os.sep + '*.tif')

    up = None
    if args.modelType == 2:
        up = torch.nn.Upsample(scale_factor=8, mode='bilinear')#定义上采样的模式,只是定义
        if args.gpu:
            up = up.cuda()

    p = args.p
    q = args.q
    classes = args.classes
    if args.modelType == 2:
        modelA = Net.UNet(3,2)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = 'model_6.pth'#3.1.1 14最好
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/encoder folder')
            exit(-1)

        modelA.load_state_dict(torch.load(model_weight_file, map_location='cpu'))

    else:
        print('Model not supported')

    if args.gpu:
        modelA = modelA.cuda()

    # set to evaluation mode
    modelA.eval()

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    evaluateModel(args, modelA, up, image_list)

    image_list_result = glob.glob(args.savedir + os.sep + 'gray_data' + os.sep + '*.png')
    nummax, z = clear_Rectangle(args, image_list_result)

    image_list_clear = glob.glob(args.savedir + os.sep + 'clear_data' + os.sep + '*.png')
    cutp = search_Rectangle(args, image_list_clear, nummax, z)

    save_draw_Rectangle(args, cutp, image_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument('--model', default="ARFNet", help='Model name')
    parser.add_argument('--data_dir', default=r"data", help='Data directory')

    parser.add_argument('--img_extn', default="tif", help='from tif change to RGB Image format')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')#程序里压根没用？
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--modelType', type=int, default=2, help='2')
    parser.add_argument('--savedir', default=r'G:/zhanghong/UNet-3-DDGF/test/results', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weightsDir', default='../pretrained/', help='Pretrained weights directory.')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier. ')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier. ')
    parser.add_argument('--cityFormat', default=False, type=bool, help='If you want to convert to cityscape '
                                                                       'original label ids')
    parser.add_argument('--colored', default=False, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--overlay', default=False, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks overlayed on top of RGB image')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    #assert (args.modelType == 1) and args.decoder, 'Model type should be 2 for ESPNet-C and 1 for ESPNet'#检查
    if args.overlay:
        args.colored = True # This has to be true if you want to overlay
    main(args)
