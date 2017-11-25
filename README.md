# DCNN

We implemented a torch module for a deformable convolution, introduced by Dai et al, see https://arxiv.org/abs/1703.06211.

For this we adapt the im2col routine, which is usually used in convolution layers.

I think the implementation is quiet faithfull, except that we interpolate points not lying in the image by projecting them onto the image for simplicity. But it should not be complicated to change this.

For some further explanations of the math and the implementation(and also some pictures) see lab_bericht.pdf.

