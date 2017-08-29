'''

2017 IFN680 W05 Prac 

Instructions: 
    - You should implement the class XYPopulation.  This class is derived from
      the abstract class 'population_search.Population' of Assignment One.
    - The rest of the script launch a particle filtering search on some multimodal 
      artificial landscape.

'''

import numpy as np
import matplotlib.pyplot as plt

import population_search

#------------------------------------------------------------------------------

class XYPopulation(population_search.Population):
    '''
        Subclass of Population specialized for cases where the individuals are 
        (x,y) points.
        
        The search problem consists in finding a global minimum of the function
        represented by the float image 'self.landscape'
        
        Usage example:

            W =  initial_population_xy([0,200,0,100], 50)    
            pop = XYPopulation(W)
            pop.set_landscape(Z)
            #pop.temperature = 1   # can experiment with this parameters
            Lw, Lc = pop.particle_filter_search(10,True)
            plt.plot(Lc)
            plt.title('Cost vs generation index')
            plt.show()
            #print(L)
            print(pop.best_w)
            print(pop.best_cost)            
            replay_search(Z,Lw)

        
    '''
    def __init__(self, W):
        '''
        Constructor. Directly pass the initial population to the parent
        class constructor.
        @param
          W : initial population
              W[i,0] is the x coord
              W[i,1] is the y coord              
        '''
        super().__init__(W)
    
    def evaluate(self):
        '''
        Evaluate the cost of each individual.
        Store the result in self.C
        That is, self.C[i] is the cost of the ith individual.
        Keep track of the best individual seen so far in 
            self.best_w 
            self.best_cost  is the corresponding best cost.
            
        '''
        # clip(a, a_min, a_max, out=None)
        height, width = self.landscape.shape[:2]
        # clip the values
        np.clip(self.W[:,0],0,width-1,self.W[:,0])
        np.clip(self.W[:,1],0,height-1,self.W[:,1])
        self.C = self.landscape[self.W[:,1].astype(int), self.W[:,0].astype(int)]
        i_min = self.C.argmin()
        cost_min = self.C[i_min]
        if cost_min<self.best_cost:
            self.best_w = self.W[i_min].copy()
            self.best_cost = cost_min
        return cost_min

    def mutate(self):
        '''
        Mutate each individual.
        Add with equal probability -1,0 or +1 to the points of self.W
        '''
        mutations = np.random.choice([-1,0,1], 2*self.n, replace=True, p = [1/3,1/3,1/3]).reshape(-1,2)  
        self.W = self.W+mutations

    def set_landscape(self, landscape):
        self.landscape = landscape

#------------------------------------------------------------------------------        

def initial_population_xy(region, pop_size=20):
    '''
        Create an initial population W by sampling point uniformly
        in the region defined by the tuple (rmx, rMx, rmy, rMy).
        
        @param
            - region : a tuple (rmx, rMx, rmy, rMy) where [rmx, rMx] is the 
                       interval from which to sample the x coords. Similary,
                       definition for the interval [rmy, rMy] and the y coords.
            - pop_size : number of points generated
        @return
            W : a pop_size by 2 array of floats.
                W[i,0] is the x coord of the ith point generated
                W[i,1] is the y coord of the ith point generated
    '''        
    # initial population
    rmx, rMx, rmy, rMy = region
    W = np.concatenate( (
                 np.random.uniform(low=rmx,high=rMx, size=(pop_size,1)) ,
                 np.random.uniform(low=rmy,high=rMy, size=(pop_size,1)) 
                        ), axis=1)    
    return W

#------------------------------------------------------------------------------        

def make_landscape(height=100, width=200):
    '''
    Make a float image from an hardcoded multi-modal function (several peaks)
    @return
        float image of 'height' rows and 'width' columns.
    '''
    X = np.repeat( 
        np.arange(width).reshape(1,-1),
        height,
        axis=0)
    Y = np.repeat( 
        np.arange(height).reshape(-1,1),
        width,
        axis=1)    
    # Could have used  np.meshgrid() !
    # Below is an arbitrary function with two valleys
    Z1 = 10*np.exp(- ((X-80)**2+(Y-40)**2)/30**2)
    Z2 = 12*np.exp(- ((X-150)**2+(Y-80)**2)/80**2)
    return -(Z1+Z2)
    
        
#------------------------------------------------------------------------------        


def replay_search(
        Z, # landscape
        L_search # list of W
                  ):
    '''
    Show how the search went.
    Each generation is replayed.
    User has to close the figure of step t in order to show step t+1
    '''

    print('Close the figure to see the next generation!')
    for i in range(len(L_search)):
        fig, ax = plt.subplots()
        plt.imshow(Z)        
        xdata, ydata = L_search[i][:,0], L_search[i][:,1]         
        ax.scatter(xdata, ydata,c='w')
        plt.title('Step {} out of {}'.format(i,len(L_search)))
        plt.show()    
    

    
#-----------------------------------------------------------------------------

# script 

Z = make_landscape()
#plt.figure()
#plt.imshow(Z)
#plt.colorbar()
#plt.title('Landscape image')
#plt.show()
   
W =  initial_population_xy([0,200,0,100], 50)  #50 for individual  
pop = XYPopulation(W)
pop.set_landscape(Z)
#pop.temperature = 1   # can experiment with this parameters
Lw, Lc = pop.particle_filter_search(20,True)  # 20 for generation Lw for location of individual
plt.plot(Lc)
plt.title('Cost vs generation index')
plt.show()
#print(L)
print(pop.best_w)
print(pop.best_cost)

replay_search(Z,Lw)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        

#plt.show()
