import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy
import matplotlib.pyplot as plt

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ
if opt.model == 'posenet':
    testepochs = numpy.arange(1, 2000+1)
else:
    testepochs = numpy.arange(1185, 1185+1)

testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch medX  medQ\n')
testfile.write('==================\n')

model = create_model(opt)
visualizer = Visualizer(opt)

for testepoch in testepochs:
    model.load_network(model.netG, 'G', testepoch)
    visualizer.change_log_path(testepoch)
    # test
    # err_pos = []
    # err_ori = []
    err = []
    err_pos = []
    err_rot = []
    print("epoch: "+ str(testepoch))
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()[0]
        print('\t%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
        image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
        pose = model.get_current_pose()
        visualizer.save_estimated_pose(image_path, pose)
        err_p, err_o = model.get_current_errors()
        # err_pos.append(err_p)
        # err_ori.append(err_o)
        err.append([err_p, err_o])
        err_pos.append(err_p)
        err_rot.append(err_o)
    total_count = len(err)
    err_pos.sort()
    err_rot.sort()
    print(err_pos)    
    
    count_list = []
    for i in numpy.arange(0, err_pos[-1], 0.01):
        count = 0
        for err_p_i in err_pos:
            if err_p_i <= i:
                count += 1
        count_list.append([i, count / total_count])
    
    count_list = numpy.array(count_list)
    numpy.save('npy/Kings_poseMultiAttn_pos', count_list)
    
    count_list = []
    for i in numpy.arange(0, err_rot[-1], 0.01):
        count = 0
        for err_r_i in err_rot:
            if err_r_i <= i:
                count += 1
        count_list.append([i, count / total_count])
    
    count_list = numpy.array(count_list)
    numpy.save('npy/Kings_poseMultiAttn_rot', count_list)
    '''
    plt.figure()
    for x, y in count_list:
        plt.scatter(x, y, c='red')
    plt.show()
    '''
        
