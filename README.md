We present an approach to the detection and
identification of human faces and describe a working, near realtime
face recognition system which recognizes the person by
comparing characteristics of the face to those of known individuals.

# Face detection
Viola Jones detector which is used for face detection
performs much better and can detect faces in real time. It can
detect faces irrespective of their scale and position. A result is
shown here by using the image of George Bush. This is not
cherry picked but just a random image on the Internet.

![detection](results/bush_output.png)

# Methods used:

## Convolutional Neural Network
We have used the architecture as shown in below table and could achieve and accuracy of 88.509%.
![Architecture](results/cnn_architecture.png)

## Independent component analysis
The top 12 ica components are shown here below.The
Image does not look as much intuitive as PCA eigen faces.But
the structure and relative position of nose, eye’s, eye brow’s,
lips etc; is still maintained
![ica](results/ica_out.png)

## Non negative matrix factorization
The highest accuracy for NMF can
achieved with projecting onto 219 components and the accuracy
is 82.9%.This is comparable to PCA and ICA but lesser
than both of them.
![nmf](results/nmf.JPG)

## Local Linear embedding
The Highest accuracy for LLE is with
number of components as 26 and number of neighbour’s as
27.The accuracy is 33.85%.It is not better than PCA, ICA
and NMF.
![lle](results/lle.JPG)

## t-Distributed Stochastic Neighbor Embedding
The highest accuracy for TSNE is
45% is almost constant with the increase in dimension’s to
which it is being projected onto.The class output is same
disregard of the input.
![tsne](results/tsne.JPG)

# Report
[Report link](https://drive.google.com/file/d/0B7AfoY1RmYCrYkJDSDlkcWk4dzA/view?usp=sharing) 
