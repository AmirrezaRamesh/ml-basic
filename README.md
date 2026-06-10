# ml models
training three simple datasets with three algorithms: perceptron, MLP, and RBF
- ML course 404-405 spring
- DR Hassanzade

### list:
- [raw data visualization](#raw-data-visualization)
- [dataset 1: two moons](#dataset-1--two-moons)
- [dataset 2: XOR](#dataset-2--xor)
- [dataset 3: circles](#dataset3--circles)
- [comparison table](#comparison-table)
- [analysis](#analysis)

### raw data visualization

![pic1](./docs/1.png)


## dataset 1 : two moons
I trained the two moons dataset created by sk-learn. here's the outputs:

### perceptron

![pic2](./docs/2.png)
![pic3](./docs/3.png)

### MLP

![pic2](./docs/4.png)
![pic3](./docs/5.png)

### RBF

![pic2](./docs/6.png)
![pic3](./docs/7.png)


## dataset 2 : XOR
I trained the two moons custom created. here's the outputs:

### perceptron

![pic2](./docs/8.png)
![pic3](./docs/9.png)

### MLP

![pic2](./docs/10.png)
![pic3](./docs/11.png)

### RBF

![pic2](./docs/12.png)
![pic3](./docs/13.png)

## dataset3 : circles

I trained the circles dataset created by sk-learn. here's the outputs:

### perceptron

![pic2](./docs/14.png)
![pic3](./docs/15.png)

### MLP

![pic2](./docs/16.png)
![pic3](./docs/17.png)

### RBF

![pic2](./docs/18.png)
![pic3](./docs/19.png)

## comparison table

| dataset | perceptron accuracy | MLP accuracy | RBF Accuracy |
|---|---:|---:|---:|
| two Moons | 0.880 | 0.987 | 0.990 |
| XOR | 0.517 | 0.872 | 0.861 |
| circles | 0.480 | 0.997 | 0.993 |

## analysis
- for circles dataset, the data of one lable is within the other one(geometry wise),and since the decision boundry for perceptron is a line, it failed very badly. checking the confusion matrix, the real value of model is even lower than the accuracy
- noisy xor dataset is also scattered quite a lot, for the same reason as last one, perceptron model output had a bad accuracy
- for mlp to have a better outcome, I added n-epochs and lowerd the learning rate, which resulted in a very accurate model
- rbf had a good accuracy for all of the datasets
- In both MLP and RBF, feature transformation helps the model solve nonlinear problems. In an MLP, the hidden layer transforms the original inputs using weights and activation functions like ReLU, tanh, or sigmoid. This allows the model to create new internal features that make complex patterns easier to separate. In an RBF network, the input features are transformed based on their distance from selected centers, usually found by K-Means. This means each data point is represented by how close it is to different centers, which helps the model separate curved or circular patterns.
- MLP learns the decision boundary by adjusting weights through training. RBF first maps data into a new distance-based feature space, then learns a simpler boundary there.