
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


 
class Support_VM():
    def __init__(self,vizualization=True):
        self.viz = vizualization
        self.colors = {1:'r',-1:'b'}
        if self.viz:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
        
    def fit(self,data):
        self.data = data
        opt_dict = {}
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for features in self.data[yi]:
                for feature in features:
                    all_data.append(feature)

        self.max_features_val = max(all_data)
        self.min_features_val = min(all_data)
        all_data = None

        step_size = [self.max_features_val * 0.1,
                     self.max_features_val * 0.01,
                     self.max_features_val * 0.001]

        b_range_multiple = 5

        b_multiple = 5
        latest_optimum = self.max_features_val*10

        for step in step_size:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_features_val*b_range_multiple),self.max_features_val*b_range_multiple
                                   ,step*b_multiple):
                    for transform in transforms:
                        w_t = w*transform
                        found_option  = True
                        for i in self.data:
                            for xi in data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >=1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0]<0:
                    optimized = True
                    print('Optimized a bit..')
                else:
                    w = w - step

            # opt_dict = {||w||:[w,b]}
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step*2




    def predict(self,new_feature):
        # x.w+b
        classification = np.sign(np.dot(np.array(new_feature),self.w)+self.b)
        if classification != 0  and self.viz:
            self.ax.scatter(new_feature[0],new_feature[1],s=200,marker='*',c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x,w,b,v):
            return (-w[0]*x - b + v) / w[1]

        datarange = (self.min_features_val*0.9,self.max_features_val*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        db1 = hyperplane(hyp_x_min,self.w,self.b,1)
        db2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])

        dbn1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        dbn2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [dbn1, dbn2])

        db01 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db02 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db01, db02])
        plt.show()






data_dict = {-1:[[1,7],
            [2,8],
            [3,8]],
            1:[[5,1],
            [6,-1],
            [7,3]]}

svm = Support_VM()
svm.fit(data=data_dict)
print(svm.predict([2,5]))
print(svm.predict([20,5]))
svm.visualize()
