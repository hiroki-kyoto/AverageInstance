# Average Instance

### Description
Average Instance is defined as the average look of what a concept can
be instantiated as. Especially, the visual concepts.

This research aims to contribute to open-set recognition, however,
it also demonstrate good ability to generate class specified random
samples with a big surprise. 

In the model proposed along with this idea, open-set classifier and
generative model are simultaneously integrated.
As an out-of-plan bonus, average instance is also proved to be
robust against adversarial attacks.
Besides, since average instances are learnt by clustering, they also
support a semi-supervised learning style.

### Guide
1. `osi_mnist.py` is the target script for validating our ideas of average
instance. So just look deep into it and you'll understand everything
concerning our work. B.T.W. `osi_mnist` means `open-set inference` in 
short. It's not called `open-set recognition` because our method
can also be used to generate unseen samples which acts like GANs, 
Flow-based models and VAEs.
2. `continual_learning_mnist.py` actually attempts to realize 
task-incremental neural network with open-set setting. However,
the overall classification performance is far below our expectation.
Thus we gave up on it and turns to other ideas to accomplish
task-incremental learning.
3. Results create by `osi_mnist.py` are stored in path `./test/`.
`./test/osi-minst/` are generated samples from generative model
in `osi_mnist.py`. Classification performances run by different
hyper-parameters are logged in `./test/osi-mnist/AE.txt`. The best
reported is as `94.0%`, which has not reached the level of SOTA.

### Future Direction
1. Figure out the reason behind the gap between softmax-based classifier
and ours.
2. Try to challenge even more difficult tasks like CIFAR-10, CIFAR-100, 
etc. On both classification and generation tasks. 