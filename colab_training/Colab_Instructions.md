# Training on Google Colab

This folder (`d:\DataSprint\colab_training`) contains all the Python files you need to train your ensemble on Google Colab. Colab's Free T4 GPU is much faster than the local GTX 1650 and has 16GB VRAM, so training all three models will be very quick!

## Step 1: Set up Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Go to **Runtime → Change runtime type**. Select **T4 GPU**, then save.

## Step 2: Upload Files to Colab

In the left sidebar of your Colab notebook, click the **Folder icon** 📁 to open the Files pane.

Upload the following files from your computer:
1. The **competition data zip file**: `C:\Users\ASUS\Downloads\train.zip`
2. The four python scripts from `d:\DataSprint\colab_training\`:
   - `train.py`
   - `dataset.py`
   - `models.py`
   - `utils.py`

*(Just drag and drop them into the Files pane. Wait for the upload circles to complete.)*

## Step 3: Run the Training!

Copy and paste the following bash commands into a single Colab cell, and run it:

```bash
# 1. Unzip the dataset quietly
!unzip -q train.zip -d /content/train

# 2. Install the model library
!pip install timm

# 3. Start the training!
!python train.py
```

## Step 4: Download Your Trained Models

Once the training finishes, the best checkpoints will be saved in the `/content/checkpoints/` folder in the Colab sidebar.

1. In the Colab Files pane, expand the `checkpoints` folder.
2. You will see three files:
   - `efficientnetv2_best.pth`
   - `resnet50_best.pth`
   - `densenet121_best.pth`
3. Hover over each file, click the **three dots** (⋮) on the right, and select **Download**.

## Step 5: Returning to Local PC

Once you've downloaded the three `.pth` files from Colab:

1. Move them into `d:\DataSprint\checkpoints\` on your local PC. *(You may need to create this folder)*.
2. Tell me: **"I have trained the models and put the checkpoints in the folder."**
3. I will then use our `predict.py` script to load your new, powerful models, run the predictions on the test set, and generate the final `submission.csv`!
