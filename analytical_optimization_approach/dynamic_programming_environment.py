import numpy as np
from matplotlib import pyplot as plt

def createRoad(object):
    def __init__(self, startpoints, endpoints, waypoints):
        self.startpoints = startpoints
        self.endpoints = endpoints
        self.waypoints = waypoints

        ## Write equation of straight lines joining the roads i.e. (m - slope, c - y intercept)
        self.roads = {}
        for i in range(len(waypoints)):
            slope = (waypoints[0][1] - waypoints[1][1])/(waypoints[0][0] - waypoints[1][0])
            intercept = waypoints[0][1] - slope*waypoints[0][0]
            self.roads[waypoints[i]] = np.array([slope,intercept])

        ## Create Roads connected to each other i.e. [a,b] is connected to [b,c] and vice versa.
        ## Both entries will be created. Also taken into account which side is the road connected
        ## to another road.
        self.connection_front = {}
        self.connection_back = {} 
        for i in range(len(waypoints)):
            temp = waypoints[:]
            ## Removes the road itself from the list to prevent conection to itself
            temp.remove(waypoints[i])

            ## Maintain directional list of connecting roads
            ## Helps in creating a unidirectional flow of traffic
            self.connection_front[waypoints[i]] = []
            self.connection_back[waypoints[i]] = []
            for wp in temp:
                if waypoints[i][0] in wp:
                    self.connection_front.append(wp)
                elif waypoints[i][1] in wp:
                    self.connection_back.append(wp)
                
                    
    ## Function to check whether custom points provided are on the roads
    def _checkPointsAreOnRoad(self, points):
        incompatiable_points = points[:]
        for p in points:
            for r in roads:
                if r[0]*p[0] + r[1] == p[1]:
                    incompatiable_points.remove(p)
                    break
        return incompatiable_points
    

    ## Create points which are roughly as per the spacing given
    def createPointsOnRoad(self, spacing):
        self.allPoints = {}
        for p in waypoints:
            x_coords = np.reshape(np.linspace(p[0][0], p[1][0], int((p[0][0] - p[1][0]) / spacing + 1)), (-1,1))
            y_coords = np.reshape(np.linspace(p[0][1], p[1][1], int((p[0][1] - p[1][1]) / spacing + 1)), (-1,1))
            self.all_points[p] = np.hstack((x_coords, y_coords))


    ## Setter Functions
    def setAllPoints(self, all_points):
        incompatiable_points = _checkPointsAreOnRoad(self, all_points)
        if len(incompatiable_points) == 0:
            self.all_points = all_points
        else:
            return incompatiable_points

    ## Getter Functions
    def getAllPoints(self):
        return {'startpoints':self.startpoints, 'endpoints':self.endpoints, 'waypoints':self.waypoints,
                'all_points':self.all_points}

    def getConnections(self):
        return {'front':self.connection_front, 'back':self.connection_back}




    
## Creates a Markov Chain which will take in the roads, the startpoints, the endpoints and other relevant
## information as the input
class createStaticMarkovChain(self):

    def __init__(self, roads, all_points, connections, connection_probabilities):
        self.roads = roads
        self.points = points
        self.connections = connections
        self.connection_probabilities = connection_probabilities # Probability to transition to one of the connecting roads at the end of a road

    def _sanityCheckConnectionProbability(self):
        for i in self.connections:
            if i not in self.connection_probabilities.keys():
                raise raise Exception("connection_probabilities has incomplete catalogue !!!")

    def createSimpleMarkovChain(self):
        ## First creat a directional Markov Chain Connecting Roads

        ## Second create using the above MC another MC with the points

        ## Trim the unreachable i.e. points that maybe reached with probability 0
        return None

        

    def initializeMarkovChain(self):
        pass


    def getNextState(self, state):
        pass

    def getCurrentState(self, state):
        return self.current_state

    
                
