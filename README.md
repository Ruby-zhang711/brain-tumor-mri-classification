# Brain Tumor MRI Classification with ResNet18

This project is an educational brain MRI image classification workflow. It uses transfer learning with ResNet18 to classify MRI images into two classes:

- `YES`: brain tumor present
- `NO`: no brain tumor

The project was prepared for internship learning and presentation. The notebook keeps the full pipeline clear and report-friendly.

## Project Highlights

- Downloads the public Kaggle brain MRI dataset
- Checks the `yes/` and `no/` image folders
- Applies data augmentation to expand the dataset
- Splits data into train, validation, and test sets
- Loads images into Python arrays
- Preprocesses images into RGB `224 x 224` format
- Uses ImageNet-pretrained ResNet18 for transfer learning
- Trains only the final classification layer
- Shows training loss and accuracy curves
- Displays prediction examples from both `YES` and `NO`
- Adds a standard TP / FP / FN / TN confusion matrix

## Folder Structure

```text
brain-tumor-mri-classification-project/
  README.md
  requirements.txt
  .gitignore
  notebooks/
    Brain_Tumor_Detection_Colab_Report.ipynb
  docs/
    Code_Explanation_Guide.md
    Code_Explanation_Guide.docx
  slides/
    Brain_Tumor_MRI_Classification_Intern_Report.pptx
```

## How to Run

The easiest way is to run the notebook in Google Colab.

1. Open Google Colab.
2. Upload `notebooks/Brain_Tumor_Detection_Colab_Report.ipynb`.
3. Run cells from top to bottom.
4. If Colab asks for package installation, allow the notebook to install the lightweight packages.
5. Use a GPU runtime if available:
   - `Runtime`
   - `Change runtime type`
   - `Hardware accelerator`
   - `GPU`

## Main Notebook

The main notebook is:

```text
notebooks/Brain_Tumor_Detection_Colab_Report.ipynb
```

It includes these main sections:

1. Environment setup
2. Imports and random seed
3. Dataset check
4. Data augmentation
5. Train / validation / test split
6. Image loading
7. Class distribution chart
8. Simple preprocessing
9. Saving processed images
10. ResNet input transforms
11. Dataset and DataLoader
12. Training function
13. ResNet18 model setup
14. Training curves
15. Prediction examples by class
16. Confusion matrix

## Notes

This is an educational project, not a clinical diagnostic system. The model and results should be used for learning and demonstration only.

