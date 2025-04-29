# LCI
This is a program for Lung Cancer Identifying by using AI.

NOT meant to be used professionally, this is only a student project.

# Notes
Noice cancellation is done after applying CLAHE, and the noise cancelled image is also getting a CLAHE applied to it. This seems to have the best clarity.  

<img src="CDNC.gif" width="300" height="300">

This image shows the result for one set of CT images.   

Creating a mask for the images will help with binarification.
Masks will be created using the original images. Goal is to remove all of the soft tissue around th ribcage and we want to keep the lungs only (or anything inside the ribcage
Images need to be converted to HU units and those can be used to create a mask.
Bones and soft tissue have higher values than lungs. This needs to be fine tuned since cancer is soft tissue and will be masked out.
  

# ToDos
- [x] Decide on noise cancellation (use before or after CLAHE)
- [ ] Create a mask of the lungs
- [ ] Binarification
- [ ] Morphology
- [ ] Edge detection
- [ ] Create a Neural Network
- [ ] Teach Network
  - [ ] Adjust hyperparameters
- [ ] Create frontend
- [ ] Connect frontend and backend
- [ ] Documentation
