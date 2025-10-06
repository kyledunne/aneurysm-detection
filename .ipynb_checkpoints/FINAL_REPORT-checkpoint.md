# Final Report

This final report is a continuation of the original analysis in the [README](README.md) file. The code for this continued analysis is in the Jupyter notebook [notebook_part_2.ipynb](notebook_part_2.ipynb) I finished the first part of the project on October 3rd, and the final report is due on October 8th, so I only had a few days to make additions - I did a lot of work for the first stage of the project, though.

Nevertheless, in these few days I want to take the opportunity to make a few upgrades to my process and model. The three upgrades I am targeting are as follows:

- Use [KerasTuner](https://keras.io/keras_tuner/) to optimize hyperparameters of my CNN
- Use and fine-tune a pre-existing image classifier as a base, that has been trained specifically for classifying medical images
- Finally, use and fine-tune a more standard image classifier as a base, such as one that has been trained on ImageNet.
- Compare the performance of the completely-from-scratch CNN, the model using a medical imaging base model, and the model using a more typical base model

I hope to complete these tasks and analyze them by the due date of the final project, October 8th. However, the final submission deadline for the *competition* is October 14th, and I will continue working on this project until (at least) that date. Any additional work will be completed in a third notebook and Markdown file to not interfere with this final report for my class in case it is in-progress when this assignment is being graded. However, if interested, feel free to check out the relevant files to see what else I am working on:

- [Report_part_3.md](Report_part_3.md)
- [notebook_part_3.md](notebook_part_3.md)

In this third part of the project, I plan to work on the following:

- Update the pipeline and model to classify a whole series of images from a single patient (as in the competition), rather than just a single image as a time
- Use all data instead of just a small subset; because the full dataset is so large, this will likely require data parallelism, using a multi-GPU VM from Google Cloud/Azure/AWS instead of Google Colab
- Update the pipeline and model to also make predictions for the 13 different regions of the brain (as required by the competition), expanding the problem beyond simple binary classification
- Look into using more advanced imaging techniques, such as object detection or even image segmentation, to attempt to improve classification performance (I think the most powerful approach would be to use a pre-existing image segmentation model that is trained on brain images to segment the image, and use that to transform the input data set into a different kind of dataset that only contains specific small regions of the images where an aneurysm could actually be, i.e., parts of the image that contain intersections of major blood vessels, and then train an image classifier on *that*.)
- Look into ensembling, i.e., using multiple different models that perform well but are at the same time as different as possible from each other, and use a weighted sum of their predictions as the final prediction.