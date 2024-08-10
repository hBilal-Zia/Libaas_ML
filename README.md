# Libaas ML
The following repository is part of Libaas (Outfit Recommendation App). It contains all the training code, links to dataset and trained models of the project.
All the models have been trained on multiple datasets using Keras and Tensorflow. The API's are developed using Flask which also handles the retrival of user data from Firebase.


# Installation & Usage
Requires google colab.
Clone Repository.
If you already have your firebase project setup get your secret key json file from your firebase project settings, if not first setup firebase project.
The API uses ngrok to make the API public. Setup your ngrok account, get your authentication token from there and set it into Libaas_API.ipynb.
Copy models to your drive, set path for every model in the Libaas_API.ipynb.
All the mdodules and 
Simply run the notebook in colab and enjoy.


# Dataset

- [Outfit-Items](https://www.kaggle.com/datasets/kritanjalijain/outfititems/data)

- [Polyvore Outfits](https://www.kaggle.com/datasets/dnepozitek/polyvore-outfits)

- [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

# References

- [Segment Body](https://github.com/TonyAssi/Segment-Body)

- [Virtual Try On](https://huggingface.co/blog/tonyassi/virtual-try-on-ip-adapter)

# Citations
```
@inproceedings{huang2017outfit,
  title={Outfit Recommendation System Based on Deep Learning},
  author={Huang, Yunan and Huang, Ting},
  booktitle={2nd International Conference on Computer Engineering, Information Science \& Application Technology (ICCIA 2017)},
  series={Advances in Computer Science Research},
  volume={74},
  year={2017},
  publisher={Atlantis Press},
  doi={10.2991/iccdata-17.2017.41}
}
```
