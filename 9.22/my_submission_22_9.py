'''
Group 9 Students: 
    LINNI QIN N9632981
    JIUZENG RONG N9814434
2017 IFN680 Assignment

Instructions: 
    - You should implement the class PatternPosePopulation

 '''

import numpy as np
import matplotlib.pyplot as plt


import pattern_utils
import population_search

#------------------------------------------------------------------------------

class PatternPosePopulation(population_search.Population):
    '''
    
    '''
    def __init__(self, W, pat):
        '''
        Constructor. Simply pass the initial population to the parent
        class constructor.
        @param
          W : initial population
        '''
        self.pat = pat
        super().__init__(W)
    
    def evaluate(self):
        '''
        Evaluate the cost of each individual.
        Store the result in self.C
        That is, self.C[i] is the cost of the ith individual.
        Keep track of the best individual seen so far in 
            self.best_w 
            self.best_cost cont
        @return 
           best cost of this generation            
        '''
 

                    
        height, width = self.distance_image.shape[:2]

        # clip the values
        np.clip(self.W[:,0],0,width-1,self.W[:,0])  #X coord
        np.clip(self.W[:,1],0,height-1,self.W[:,1])  #Y coord
        
        
        
        for i in range(self.n):
            #variable indiv_cost is a tuple storing (cost, three position of vertices in array) of each random population
            #self.pat is an object of pattern_utils and it is used to call evaluate method in pattern_utils.py
            #self.distance_image is cost matrix given by the 2D float array 'imd'
            #Therefore we call evalute method to calculate three segments distance between (self.distance_image) and/n
            #each triangle (self.W[i])
            indiv_cost = self.pat.evaluate(self.distance_image,self.W[i]) 
            # assign first element of indiv_cost which is the cost of each population in self.C
            self.C[i] = np.array(indiv_cost[0])
        #search index of the minimun of self.C and assign it in variable i_min
        i_min = self.C.argmin()
        #assign the munimun cost value in variable cost_min
        cost_min = self.C[i_min]
        
        # compare current cost_min with previous cost_min (variable self.best_cost),
        # if cost_min is smaller, assign it in self.best_cost
        if cost_min<self.best_cost:
            #assign four features of best individual in self.best_w
            self.best_w = self.W[i_min].copy()
            #assign cost_min in self.best_w
            self.best_cost = cost_min
        return cost_min

    def mutate(self):
        '''
        Mutate each individual.
        The x and y coords should be mutated by adding with equal probability 
        -1, 0 or +1. That is, with probability 1/3 x is unchanged, with probability
        1/3 it is decremented by 1 and with the same probability it is 
        incremented by 1.
        The angle should be mutated by adding the equivalent of 1 degree in radians.
        The mutation for the scale coefficient is the same as for the x and y coords.
        @post:
          self.W has been mutated.
        '''
        
        assert self.W.shape==(self.n,4)

        # mutations is two demensions array with (population_size, four features), four features: x coord, y coord, theta,scale
        # each feature mutates -1, 0 or 1 randomly with equal posibility 1/3 and changes its type to float (explain later)
        mutations = np.random.choice([-1,0,1], 4*self.n, replace=True, p = [1/3,1/3,1/3]).reshape(-1,4).astype(float)
        # as the third element is theta and each iteration should mutate 1 degree
        # 1 degree is equal to 0.01745 radians and its type should be float otherwise the relust would be 0 all the time
        mutations[:,2]*= 0.01745
        # mutate all population
        self.W = self.W + mutations
                
    def set_distance_image(self, distance_image):
        self.distance_image = distance_image

#------------------------------------------------------------------------------        

def initial_population(region, scale = 10, pop_size=20):
    '''
    
    '''        
    # initial population: exploit info from region
    rmx, rMx, rmy, rMy = region
    W = np.concatenate( (
                 np.random.uniform(low=rmx,high=rMx, size=(pop_size,1)) ,
                 np.random.uniform(low=rmy,high=rMy, size=(pop_size,1)) ,
                 np.random.uniform(low=-np.pi,high=np.pi, size=(pop_size,1)) ,
                 np.ones((pop_size,1))*scale
                 #np.random.uniform(low=scale*0.9, high= scale*1.1, size=(pop_size,1))
                        ), axis=1)    
    return W

#------------------------------------------------------------------------------        
def test_particle_filter_search():
    '''
    Run the particle filter search on test image 1 or image 2of the pattern_utils module
    
    '''
    
    if True:
        # use image 1
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_1(True)
        ipat = 2 # index of the pattern to target
    else:
        # use image 2
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(True)
        ipat = 0 # index of the pattern to target
        
    # Narrow the initial search region
    pat = pat_list[ipat] #  (100,30, np.pi/3,40),
    # print(pat) 
    xs, ys = pose_list[ipat][:2]  #(100, 30, 1.0471975511965976, 40)
    region = (xs-20, xs+20, ys-20, ys+20)
    scale = pose_list[ipat][3]
    
    
    
    
    
    
    ##counter = 0  # counter bad cost which value greater than 1 to calculate accuracy rate
    ##for i in range(30):  #repeat 30 times to  test 
    pop_size=70
    W = initial_population(region, scale , pop_size)
    
    pop = PatternPosePopulation(W, pat)
    pop.set_distance_image(imd)
    
    pop.temperature = 5


    Lw, Lc = pop.particle_filter_search(55,log=True)  #Lw each individual(x,y,theta,scale)
    
    
    
    plt.plot(Lc)
    plt.title('Cost vs generation index')
    plt.show()
    
    print(pop.best_w)
    ##choose best_cost which value great than 1 as bad cost
    ##if pop.best_cost > 1:
        ##counter += 1
    print(pop.best_cost)
    ##print("-"*40+str(counter))
    
        
    pattern_utils.display_solution(pat_list, 
                      pose_list, 
                      pat,
                      pop.best_w)
                      
    pattern_utils.replay_search(pat_list, 
                      pose_list, 
                      pat,
                      Lw)
    
#------------------------------------------------------------------------------        

if __name__=='__main__':
    test_particle_filter_search()
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
#        
#    def test_2():
#        '''
#        Run the particle filter search on test image 2 of the pattern_utils module
#        
#        '''
#        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(False)
#        pat = pat_list[0]
#        
#        #region = (100,150,40,60)
#        xs, ys = pose_list[0][:2]
#        region = (xs-20, xs+20, ys-20, ys+20)
#        
#        W = initial_population_2(region, scale = 30, pop_size=40)
#        
#        pop = PatternPosePopulation(W, pat)
#        pop.set_distance_image(imd)
#        
#        pop.temperature = 5
#        
#        Lw, Lc = pop.particle_filter_search(40,log=True)
#        
#        plt.plot(Lc)
#        plt.title('Cost vs generation index')
#        plt.show()
#        
#        print(pop.best_w)
#        print(pop.best_cost)
#        
#        
#        
#        pattern_utils.display_solution(pat_list, 
#                          pose_list, 
#                          pat,
#                          pop.best_w)
#                          
#        pattern_utils.replay_search(pat_list, 
#                          pose_list, 
#                          pat,
#                          Lw)
#    
#    #------------------------------------------------------------------------------        
#        
    