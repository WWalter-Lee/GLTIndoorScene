# GLTScene
The code for our paper [GLTScene: Global‐to‐Local Transformers for Indoor Scene Synthesis with General Room Boundaries](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15236) (PG 2024).

We present GLTScene, a novel data-driven method for high-quality furniture layout synthesis with general room boundaries as conditions. This task is challenging since the existing indoor scene datasets do not cover the variety of general room boundaries. We incorporate the interior design principles with learning techniques and adopt a global-to-local strategy for this task. Globally, we learn the placement of furniture objects from the datasets without considering their alignment. Locally, we learn the alignment of furniture objects relative to their nearest walls, according to the alignment principle in interior design. The global placement and local alignment of furniture objects are achieved by two transformers respectively. 

![teaser](README.assets/teaser.png)

# Installation & Dependencies

The dependencies of our code have been recorded in `environment.yaml`. You can create a conda environment called `GLTScene` using

```
conda env create -f environment.yaml
```

# Dataset 

We used the [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) and the [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) dataset to train our model. To download both datasets, please follow the guidelines outlined on the dataset's [webpage](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset).

## Data Preprocessing

After downloading 3D-FRONT and 3D-FUTURE datasets, you need to run the `preprocess_data.py` script to prepare data for training models.

```
python preprocess_data.py 
path_to_config
path_to_output_dir 
path_to_3d_front_dataset_dir 
path_to_3d_future_dataset_dir 
path_to_3d_future_model_info 
path_to_floor_plan_texture_images
--annotation_file config_file
--dataset_filtering threed_front_bedroom
```

For example:

```
python preprocess_data.py 
../config/bedrooms_config.yaml
../data/bedroom
../../../data/3D-FRONT
../../../data/3D-FUTURE-model
../../../data/3D-FUTURE-model/model_info.json
../demo/floor_plan_texture_images
--annotation_file ../config/bedroom_threed_front_splits.csv
--dataset_filtering threed_front_bedroom
```

For living and dining rooms, adjust the room size to 6.2m from the default 3.1m using the --room_side argument during rendering.

Please note that the initial run of `preprocess_data.py` takes time to get the `threed_front.pkl` file with data for all rooms in the scenes. After obtaining it, you can speed up the script by uncommenting `path_to_scene` in `\GLTScene\scene_synthesis\datasets\threed_front.py`(line 179).

## Pickle the 3D-FUTURE dataset

Most of the scripts need a path to a pickled `ThreedFutureDataset` file. Use  `pickle_threed_future_dataset.py` to generate it:

```
python pickle_threed_future_dataset.py 
path_to_output_dir 
path_to_3d_front_dataset_dir 
path_to_3d_future_dataset_dir 
path_to_3d_future_model_info 
--annotation_file config_file
--dataset_filtering room_type
```

# Usage

## Training

```
python train_network.py path_to_config_yaml path_to_output_dir
```

## Inference

```
python generate_scenes.py 
path_to_config_yaml 
path_to_output_dir 
path_to_3d_future_pickled_data 
path_to_floor_plan_texture_images 
--weight_file path_to_weight_file
```

# Citation

If you find this useful for your research, please consider citing:

```
@inproceedings{li2024gltscene,
  title={GLTScene: Global-to-Local Transformers for Indoor Scene Synthesis with General Room Boundaries},
  author={Li, Yijie and Xu, Pengfei and Ren, Junquan and Shao, Zefan and Huang, Hui},
  booktitle={Computer Graphics Forum},
  pages={e15236},
  year={2024},
  organization={Wiley Online Library}
}
```
