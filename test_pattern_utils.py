

import numpy as np
import matplotlib.pyplot as plt


import pattern_utils

imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(True)

#
#p1 = pattern_utils.Square()
#p2 = pattern_utils.Triangle(2)
#
#print(p1)
#print(p2)
#
#imf = pattern_utils.pat_image(
#                  [p1, p1, p2],
#                 [(10,20,np.pi/6,20), (50,30,0,30), (100,30, -np.pi/3,50)])
#imd = pattern_utils.dist_image(imf)
#
#plt.figure()
#plt.imshow(imf)
#plt.title('Pattern image')
#plt.figure()
# 
#plt.imshow(imd, cmap='hot')
#plt.title('Distance image')
#plt.colorbar()
#
#plt.show()

#best, L = pattern_utils.ce_search(imd,
#                        p1,
#                        (50,90,30,60))
#
#print(L)
#Ls = [s for s,_ in L]
#plt.figure()
#plt.plot(Ls)
#print(best)
#plt.show()
#
#test_1()

#    zz = np.arange(36).reshape(3,4,3)
#    print(zz[:,:,0])
#    X,Y,S =  pattern_utils.scan_segment((0,0),(3,2),zz)
#    print(X,Y,S,sep='\n')