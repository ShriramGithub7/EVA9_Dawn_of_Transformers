# EVA9_Dawn_of_Transformers

## Problem Statement: 


Build the following network: <\n>
<\n>
- That takes a CIFAR10 image (32x32x3)
- Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
- Apply GAP and get 1x1x48, call this X
- Create a block called ULTIMUS that:
  - Creates 3 FC layers called K, Q and V such that:
    - X*K = 48*48x8 > 8
    - X*Q = 48*48x8 > 8 
    - X*V = 48*48x8 > 8 
  - then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
  - then Z = V*AM = 8*8 > 8
  - then another FC layer called Out that:
    - Z*Out = 8*8x48 > 48
- Repeat this Ultimus block 4 times
- Then add final FC layer that converts 48 to 10 and sends it to the loss function.
- Model would look like this C>C>C>U>U>U>U>FFC>Loss
- Train the model for 24 epochs using the OCP that I wrote in class. Use ADAM as an optimizer. 
- Submit the link and answer the questions on the assignment page:
  - Share the link to the main repo (must have Assignment 7/8/9 model7/8/9.py files (or similarly named))
  - Share the code of model9.py
  - Copy and paste the Training Log
  - Copy and paste the training and validation loss chart

## Solution:
- Created model as per the given specification and code of the model (model9.py) file is present in model folder of below repo
  https://github.com/ShriramGithub7/CNN-Master

- Model is trained and below is training logs

EPOCH: 1
Batch_id=781 Loss=nan Accuracy=10.05: 100%|██████████| 782/782 [00:30<00:00, 25.55it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 2
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.68it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 3
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:27<00:00, 28.69it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 4
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.74it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 5
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.45it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 6
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.48it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 7
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.12it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 8
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.69it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 9
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.62it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 10
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.62it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 11
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:25<00:00, 30.16it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 12
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.68it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 13
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.74it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 14
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.39it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 15
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.93it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 16
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.43it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 17
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:25<00:00, 30.09it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 18
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.02it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 19
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.16it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 20
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.18it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 21
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.46it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 22
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.47it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 23
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.53it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

EPOCH: 24
Batch_id=781 Loss=nan Accuracy=10.00: 100%|██████████| 782/782 [00:26<00:00, 29.43it/s]
Test set: Average loss: nan, Accuracy: 1000/10000 (10.00%)

## Conclusion:
- Model code needs to be analyzed and changed as its accuracy is not increasing beyond 10%.
