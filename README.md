# DEEM
The codes for "context-guided entropy minimization for semi-supervised domain adaptation". The work can be downloaded from [here](https://doi.org/10.1016/j.neunet.2022.07.011).

## Env setting
    Python=3.8
    Pytorh=1.7.0 (py3.8_cuda11.0.221)
    torchvision=0.8.1  

## Dataset setting
 1. All datasets can be seen in [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [office-home](https://www.hemanthdv.org/officeHomeDataset.html), [multi (DomainNet126)](http://csr.bu.edu/ftp/visda/2019/multi-source/). After these dataset are downloaded, please bulid a new folder named ./data/ and put the dataset into it.
2. To specify your dataset path, please set "project_root" in return_dataset.py

## Training usages
Training on Office-31:

        python main.py --dataset Office-31 --s webcam --t amazon --gpu_id 0 --train 1
Test on Office-31:

        python main.py --dataset Office-31 --s webcam --t amazon --gpu_id 0 --train 0

The other datasets follow the similar usages!

If you find the repo is helpful, feel free to star and cite us:

    @article{MA2022270,
    title = {Context-guided entropy minimization for semi-supervised domain adaptation},
    journal = {Neural Networks},
    volume = {154},
    pages = {270-282},
    year = {2022},
    issn = {0893-6080},
    doi = {https://doi.org/10.1016/j.neunet.2022.07.011},
    url = {https://www.sciencedirect.com/science/article/pii/S0893608022002672},
    author = {Ning Ma and Jiajun Bu and Lixian Lu and Jun Wen and Sheng Zhou and Zhen Zhang and Jingjun Gu and Haifeng Li and Xifeng Yan},
    }
  


   
    
