#SUPPORT VECTOR MACHINE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)


    #train
    def fit(self, data):
        self.data = data
        opt_dict = {}
        
        transforms = [
            [1,1],
            [1,-1],
            [-1,1],
            [-1,-1]
        ]

        all_data = []
        for yi in self.data:                            # yi = 1 / -1
            for featureset in self.data[yi]:            # featureset = [1,7], ...
                for feature in featureset:              # feature = 1 / 7 / ...
                    all_data.append(feature)

        self.max_feature_value = max(all_data)          #   8
        self.min_feature_value = min(all_data)          #  -1
        all_data = None

        step_sizes = [
            self.max_feature_value * 0.1,               #  0.8
            self.max_feature_value * 0.01,
            #point of expense
            self.max_feature_value * 0.001,
        ]
        # extremely expensive
        b_range_multiple = 5

        b_multiple = 5
        latest_optimum = self.max_feature_value*10      #   80

        for step in step_sizes:                                 # 0.8, 0.08, ...
            w = np.array([latest_optimum,latest_optimum])       # [80 80] for all steps
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), 
                                    self.max_feature_value*b_range_multiple,
                                    step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    print("Optimized a step.")
                else:
                    w = w-step
            
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step


    def predict(self, features):
        classification = np.sign(np.dot(np.array(features),self.w) + self.b)
        if classification != 0  and self.visualization:
            self.ax.scatter(features[0],features[1], s=200, marker="*", c=self.colors[classification])
        
        return classification

    # purely for visualizing, machine doesn't need.
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # v = x.w+b
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]
        
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # w.x+b = 1
        # positive support vector
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max], [psv1,psv2])

        # w.x+b = -1
        # negative support vector
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2])

        # w.x+b = 0
        # decision boundary support vector
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max], [db1,db2])

        plt.show()



data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),

              1:np.array([[5,1],
                          [6,-1],
                          [7,3],]) }

svm = Support_Vector_Machine()
svm.fit(data=data_dict)
svm.visualize()