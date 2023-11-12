# Insights

## Forward Diffusion
From implementing forward diffusion, I realized (1) that it can be computationally expensive to sample a large multivariate normal distribution without a GPU and (2) normalizing the input image (e.g., dividing the initial RGB image by 255 and every clipping every $x_t$ between $[0, 1]$ dramatically improves the degenerative process; otherwise, the image may just go entirely black) 