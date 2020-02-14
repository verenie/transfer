# transfer
Convolutional Neural Network architecture transfer learning in data starved regimes with model fine-tuning and convolutional feature extraction.

# train custom models in data starved environment
To reproduce with a custom dataset follow directions below:

1. Install PyTorch
2. Clone or download code in your working directory
3. Provide a data folder where each inner folder represents image classes.
4. Use datax or dataxm to process data into appropriate train/val/test sets by specifying the following:
```python
OUTPUT_DIR = "./data/tel"
# class map keys corresspond to image classes, i.e. dog/cat/monkey
class_map = {'alpha': "./data/raw_tel/tels", 'beta': "./data/raw_tel/not_tels" }
dir_type = ['val','test','train']

# settings below will perform a TRAIN/VAL/TEST split of 60/20/20 percent
# perform two splits using the train_test_split method from sklearn
# first split at 80/20 and second on 75/25 on the remaining train set
TEST_SIZE = 0.25
VAL_SIZE = 0.2
```
This will produce a single (datax) or multiple folds (datax_m) of your data in appropriate train/val/test folders. The class_map structures maps your desired class names (alpha,beta) to folder locations in your raw data folder. The integer SEED value can be set to any integer and is used to enforce consistency of the test set across folds.

5. Select any of the five architectures implemented as PyTorch modules and specify the data directory as such:
```python
DATA_DIR = 'data/tel' # whatever your direcotry is
```
6. Specify a variant of the selected architecture (densenet has several variants of different depth) as follows:
```python
model_conv = torchvision.models.densenet161(pretrained=True)
```
7. Ensure that your final classification layer is set properly, here is a two class example for densenet, again each architecture will differ in its final layer, see the implementations provided:
```python
 # set new layer (densenet final layer is classifier)
 num_features = model_conv.classifier.in_features
 model_conv.classifier = nn.Linear(num_features,2)
 ``` 
8. Select between whole model fine-tuning or feature extraction by setting the FREEZE flag at the top of the script:
```python
# freeze layers or finetune the whole model
FREEZE = True
```
9. Specify a model name:
```python
    torch.save(model_conv,'models/yourmodelname.pt')
```
10. Run the file to retrain/finetune a new model based on your data.  NOTE: you may wish to increase the number of epochs depending on the size of your dataset.  The model training and selection process will iterate for X number of epochs and return the best performing model selected either for ACCURACY or SENSITIVITY for the class selected.

# test pretrained TEL models
Test experimental models trained on the original TEL dataset against the OOD dataset or your own images.

1. Select the testnetwork.py module.
2. Specify the data directory that follows conventions described above or download the OOD dataset and unzip in your working directory.
3. Update the tranformations to 299 for the Inception architecture or 224 for other architectures as follows:
```python
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        #transformsx.GausNoise(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])
}
```
4. Specify the model location and name:
```python
MODEL_NAME = 'models/inception3TEL.pt'
```
5. Save and run the file.


