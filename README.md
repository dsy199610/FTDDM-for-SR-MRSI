# A Flow-based Truncated Denoising Diffusion Model for super-resolution Magnetic Resonance Spectroscopic Imaging (Medical Image Analysis)

Siyuan Dong, Zhuotong Cai, Gilbert Hangel, Wolfgang Bogner, Georg Widhalm, Yaqing Huang, Qinghao Liang, Chenyu You, Chathura Kumaragamage, Robert K Fulbright, Amit Mahajan, Amin Karbasi, John A Onofrey, Robin A de Graaf, James S Duncan

[[Paper Link](https://authors.elsevier.com/a/1jr%7EK4rfPmHr0I)]

### Citation
If you use this code please cite:

	@article{dong2024flow,
	  title={A Flow-based Truncated Denoising Diffusion Model for super-resolution Magnetic Resonance Spectroscopic Imaging},
	  author={Dong, Siyuan and Cai, Zhuotong and Hangel, Gilbert and Bogner, Wolfgang and Widhalm, Georg and Huang, Yaqing and Liang, Qinghao and You, Chenyu and Kumaragamage, Chathura and Fulbright, Robert K and others},
	  journal={Medical Image Analysis},
	  pages={103358},
	  year={2024},
	  publisher={Elsevier}
	}
   
### Environment and Dependencies
 Requirements:
 * python 3.7.11
 * pytorch 1.1.0
 * pytorch-msssim 0.2.1
 * torchvision 0.3.0
 * numpy 1.19.2

### Directory
    train.py                            # main file for training FTDDM
    test.py                       	# main file for testing FTDDM
    FTDDM_MultiRes
    ├──train_flow.py                    # training normalizing flow model
    ├──cInvNet_MRI_LearnablePrior.py    # normalizing flow model
    ├──MRSI_dataset.py                  # dataloader for diffusion model
    ├──MRSI_dataset_flow.py             # dataloader for flow model
    ├──utils
    ├  ├──logs.py                       # logging
    ├  └──utils.py                      # utility files
    ├──gaussian_diffusion.py	        # create diffusion model object
    └──other files contain supporting functions for diffusion model

This code is modified based on [[Improved DDPM](https://github.com/openai/improved-diffusion)]
