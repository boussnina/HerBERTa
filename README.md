

# HerBERT - An information extraction pipeline for NHMD-Herbarium-sheets

HerBERT is an information extreaction pipeline for high resolution images owned by the Natural Museum of Denmark. The 

## Requirements 
To setup and run the pipeline, follow the steps below: 

## Setup a virtual environment and download requirements
1) Create a local folder on your machine and download the repository into the folder 

2) Create a virtual environment in the directory of the folder
```python3 -m venv venv```

3) Activate virtual environment
```source venv/bin/activate```
 
4) Install requirements via pip into virtual environment
```pip install -r requirements.txt```


5) Install the package of this GitHub repository `https://github.com/NHMDenmark/HTROCR`

```pip install git+https://github.com/NHMDenmark/HTROCR.git@package#egg=htrocr```

6) Download nmhd_base_full_final, transformer_mixed_data_set and label_detector.pt from the link provided on the thesis' frontpage
- nhmd_full_base_final: Put this folder into the HerBERT folder
- transformer_mixed_data_set: Put this folder into the MODELS folder
- label_detector.pt: Put this model into the WEIGHTS folder


## Main function of the pipeline

The main function of the program executes the program. If the user provides an imagepath to the commandline 
```--imagepath /path/to/image/```,  an example image has been provided in the repository (sp62978708987886213540.jpg),
the pipeline detects the entities ofthe input image. If the user does not provide an imagepath as a commandline argument, 
the pipeline downloads a user defined amount of images from 
the nhmd-database-server and detects on the random selection. 
In both cases the output is saved into a csv-file located in the OUTPUT directory.

```python
def main(self, image_path=None):
        """
        Main function to run the pipeline.

        Args:
            image_path (str, optional): Path to the input image. If not provided, a random image will be downloaded.
        """
        if image_path:
            self.update_database(select_random_image=False, image_path=image_path)
        else:
            self.update_database(select_random_image=True, download_amount=1)
```

# Run The program 

Execute `python3 main.py` to run the pipeline on a randomly selected image from the NHMD collection. 

Execute `python3 main.py --imagepath /path/to/image/` to run the pipeline on a specific image.




