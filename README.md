# webly-supervised
**Legacy** code recovered from the project done for the webly-supervised network paper.

The code is divided into two parts:

1. `layers` folder contains the loss function implemented to encode the relationship graph. Specifically, it computes the cross entropy between the soft-max output and the smoothed target category.

2. `subdiscover` folder contains the code used to extract `fc7` features for subcategory discovery -- finding bounding boxes of a category given weak, noisy (that is, webly) supervision of category labels for the entire image.