
#include <TH.h>
#include <luaT.h>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <math.h>
#include <assert.h> 
#include <random>
#include <stdexcept>
#include <iostream>


#include "deformableconvolution.h"


/* @brief Helper function for bilinearInterp:
 * projects a coordinate x to 0 if x < 0 and to range if x > range
 */
double projection(double x, std::size_t range){
    if(x < 0)
        return 0;
    if(x > range)
        return double(range);
    return x;
}

/* @brief interpolation using the formula (3) + (4) from [1].
 * We project the point (px,py) onto the image, interpolate the projected point
 * (px_projected,py_projected) and then multiply by 
 * (1-abs(px-px_projected))(1-abs(py-py_projected)) to correct.
 * @param input 3D-Tensor representation of an image
 * @param c the image channel we want to use
 * @params py,px the (fractional) (y,x)-coordinates where we want to interpolate
 * @return retval the interpolated value of image(py,px)
 */
static double bilinearInterp(THDoubleTensor* input, THLongTensor* bi, THDoubleTensor* bw, long c, double py, double px, std::size_t i, std::size_t j, long save_buffer){
    std::size_t H = THDoubleTensor_size(input, 1); // input image height
    std::size_t W = THDoubleTensor_size(input, 2); // input image width
    
    double *data = THDoubleTensor_data(input);
    
    std::size_t s0 = THDoubleTensor_stride(input, 0);
    std::size_t s1 = THDoubleTensor_stride(input, 1);
    std::size_t s2 = THDoubleTensor_stride(input, 2);
    
    
    double retval = 0;
    
    if((px <= -1) or (px >= W) or (py <= -1) or (py >= H)){
        return retval;
    }
    
    double px_projected = projection(px, W-1);
    double py_projected = projection(py, H-1);
    
    long ax = long(floor(px_projected)); 
    long ay = long(floor(py_projected));
    
    if(ax == W-1)
        --ax;
    if(ay == H-1)
        --ay;
    
    //dim(bw) = 4 x c1*kH*kW x H_{out}*W_{out}
    double w0,w1,w2,w3;
    
    w0 = (1-std::abs(py_projected - ay))*(1-std::abs(px_projected-ax));
    w1 = (1-std::abs(py_projected - ay))*(1-std::abs(px_projected-(ax+1)));
    w2 = (1-std::abs(py_projected - (ay+1)))*(1-std::abs(px_projected-ax));
    w3 = (1-std::abs(py_projected - (ay+1)))*(1-std::abs(px_projected-(ax+1)));

    
    
    if(px_projected != px or py_projected != py){
        w0 *= (1-std::abs(px-px_projected))*(1-std::abs(py-py_projected));
        w1 *= (1-std::abs(px-px_projected))*(1-std::abs(py-py_projected));
        w2 *= (1-std::abs(px-px_projected))*(1-std::abs(py-py_projected));
        w3 *= (1-std::abs(px-px_projected))*(1-std::abs(py-py_projected));
    }
    
    retval = w0*data[s0*c+s1*ay+s2*ax]
            +w1*data[s0*c+s1*ay+s2*(ax+1)]
            +w2*data[s0*c+s1*(ay+1)+s2*ax]
            +w3*data[s0*c+s1*(ay+1)+s2*(ax+1)];
    
    if(save_buffer){
        long *bi_data = THLongTensor_data(bi);
        double *bw_data = THDoubleTensor_data(bw);
        
        std::size_t t0 = THDoubleTensor_stride(bw, 0);
        std::size_t t1 = THDoubleTensor_stride(bw, 1);
        std::size_t t2 = THDoubleTensor_stride(bw, 2);
    
        std::size_t r0 = THLongTensor_stride(bi, 0);
        std::size_t r1 = THLongTensor_stride(bi, 1);
        std::size_t r2 = THLongTensor_stride(bi, 2);
        
        bi_data[r0*i+r1*j+r2*0] = c;
        bi_data[r0*i+r1*j+r2*1] = ay;
        bi_data[r0*i+r1*j+r2*2] = ax;
        
        bw_data[t0*i + t1*j+t2*0] = w0;
        bw_data[t0*i + t1*j+t2*1] = w1;
        bw_data[t0*i + t1*j+t2*2] = w2;
        bw_data[t0*i + t1*j+t2*3] = w3;
    }
    
    return retval;
}

/* @brief lua-wrapper for the function bilinearInterp
 * @param L lua_State which contains the parameters 3D-input-image, input channel, interpolation point parameters py, px
 * @return 1 hands the bilinear Interpolation retval of image(c,py,px) back to lua_State
 */
static int bilinearInterpolation(lua_State *L){
    THDoubleTensor *image = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
    THLongTensor *bi = (THLongTensor*)luaT_checkudata(L, 2, "torch.LongTensor");
    THDoubleTensor *bw = (THDoubleTensor*)luaT_checkudata(L, 3, "torch.DoubleTensor");
    std::size_t c = luaL_checkinteger(L, 4)-1; 
    double py = luaL_checknumber(L, 5)-1;
    double px = luaL_checknumber(L, 6)-1;
    std::size_t i = luaL_checkinteger(L, 7)-1;
    std::size_t j = luaL_checkinteger(L, 8)-1;
    long save_buffer = luaL_checkinteger(L, 9);
    
    double retval = bilinearInterp(image, bi, bw, c, py, px, i, j, save_buffer);
    
    lua_pushnumber(L, retval);
    return 1;
}

static int im2colold(lua_State *L)
{
    THDoubleTensor *image = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
    std::size_t kH = luaL_checkinteger(L, 2); 
    std::size_t kW = luaL_checkinteger(L, 3);
    std::size_t C = THDoubleTensor_size(image, 0); //number of input channels
    std::size_t H = THDoubleTensor_size(image, 1); //input image height
    std::size_t W = THDoubleTensor_size(image, 2); // input image width
    
    std::size_t wOutputImage = W-kW+1;
    std::size_t hOutputImage = H-kH+1;
    THDoubleTensor* outputImage = THDoubleTensor_newWithSize2d(kW*kH*C, wOutputImage*hOutputImage);
    std::size_t s0 = THDoubleTensor_stride(outputImage, 0);
    std::size_t s1 = THDoubleTensor_stride(outputImage, 1);
    std::size_t t0 = THDoubleTensor_stride(image, 0);
    std::size_t t1 = THDoubleTensor_stride(image, 1);
    std::size_t t2 = THDoubleTensor_stride(image, 2);
    std::size_t c,k,l,i,j;
    
    double* odata = THDoubleTensor_data(outputImage);
    double* idata = THDoubleTensor_data(image);
    
    for(c = 0; c <= C-1; c++){
        for (l = 0; l <= kH - 1; l++){
            for (k = 0; k <= kW - 1 ; k++){
                for (i = 0; i <= hOutputImage - 1; i++){
                    for (j = 0; j <= wOutputImage - 1; j++){
                        odata[(c*kW*kH+l*kW+k)*s0+s1*(i*wOutputImage+j)] = idata[(c)*t0+(i+l)*t1+(j+k)*t2];
                    }
                }
            }
        }
    }
    luaT_pushudata(L, outputImage, "torch.DoubleTensor");

    return 1;
}

/* @brief turns 3D-batches from the 3D-input-image into columns in the 2D-output-matrix,
 * taking into account offset data
 * @param L lua_State which contains the input image, the offset data, and batch size parameters kH and kW
 * @return 1 hands im2col(input) back to the lua_State
 */
static int im2col(lua_State *L)
{
    THDoubleTensor *image = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor"); //dimension(image) = C x H x W
    THDoubleTensor *offsets = (THDoubleTensor*)luaT_checkudata(L, 2, "torch.DoubleTensor"); //dimension(offsets) = H_out x W_out x kH x kW x 2
    std::size_t kH = luaL_checkinteger(L, 3); //kernel height
    std::size_t kW = luaL_checkinteger(L, 4); //kernel width
    THLongTensor *bi = (THLongTensor*)luaT_checkudata(L, 5, "torch.LongTensor");
    THDoubleTensor *bw = (THDoubleTensor*)luaT_checkudata(L, 6, "torch.DoubleTensor");
    long save_buffer = luaL_checkinteger(L, 7);
    
    std::size_t C = THDoubleTensor_size(image, 0); //number of input channels
    std::size_t H = THDoubleTensor_size(image, 1); //input image height
    std::size_t W = THDoubleTensor_size(image, 2); //input image width
    
    std::size_t a = THDoubleTensor_size(offsets, 0); //hOutputImage
    std::size_t b = THDoubleTensor_size(offsets, 1); //wOutputImage
    std::size_t ce = THDoubleTensor_size(offsets, 2); //kH
    std::size_t d = THDoubleTensor_size(offsets, 3); //kW
    std::size_t e = THDoubleTensor_size(offsets, 4); //2    
    
    std::size_t wOutputImage = W-kW+1;
    std::size_t hOutputImage = H-kH+1;
    
    std::cout << C << " " << H << " " << W << std::endl;
    THDoubleTensor* outputImage = THDoubleTensor_newWithSize2d(kW*kH*C, wOutputImage*hOutputImage); //im2col(input)
    
    std::size_t s0 = THDoubleTensor_stride(outputImage, 0);
    std::size_t s1 = THDoubleTensor_stride(outputImage, 1);
    
    std::size_t t0 = THDoubleTensor_stride(offsets, 0);
    std::size_t t1 = THDoubleTensor_stride(offsets, 1);
    std::size_t t2 = THDoubleTensor_stride(offsets, 2);
    std::size_t t3 = THDoubleTensor_stride(offsets, 3);
    std::size_t t4 = THDoubleTensor_stride(offsets, 4);
    
    std::size_t r0 = THDoubleTensor_stride(image, 0);
    std::size_t r1 = THDoubleTensor_stride(image, 1);
    std::size_t r2 = THDoubleTensor_stride(image, 2);
    
    long c;
    std::size_t k,l,i,j;
    
    double* odata = THDoubleTensor_data(outputImage);
    double* idata = THDoubleTensor_data(image);
    double* offsets_data = THDoubleTensor_data(offsets);
    double x_offset = 0;
    double y_offset = 0;
    
    for(c = 0; c <= C - 1; c++){
        for (i = 0; i <= hOutputImage - 1; i++){
            for (j = 0; j <= wOutputImage - 1; j++){
                for (l = 0; l <= kH - 1; l++){
                    for (k = 0; k <= kW - 1 ; k++){
                        assert(i < a);
                        assert(j < b);
                        assert(l < ce);
                        assert(k < d);
                        assert(1 < e);
                        y_offset = offsets_data[t0*i+t1*j+t2*l+t3*k+t4*0];
                        x_offset = offsets_data[t0*i+t1*j+t2*l+t3*k+t4*1];
                        odata[(c*kW*kH+l*kW+k)*s0+s1*(i*wOutputImage+j)] = bilinearInterp(image, bi, bw, c, i+l+y_offset, j+k+x_offset,c*kW*kH+l*kW+k,i*wOutputImage+j,save_buffer);
                        assert(odata[(c*kW*kH+l*kW+k)*s0+s1*(i*wOutputImage+j)] == idata[(c)*r0+(i+l)*r1+(j+k)*r2]);
                    }
                }
            }
        }
    }
    luaT_pushudata(L, outputImage, "torch.DoubleTensor");

    return 1;
}


static int update_grad_input(lua_State *L){
    THDoubleTensor *gradInput = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
    THDoubleTensor *gradIm2col = (THDoubleTensor*)luaT_checkudata(L, 2, "torch.DoubleTensor");
    THLongTensor *bi = (THLongTensor*)luaT_checkudata(L, 3, "torch.LongTensor");
    THDoubleTensor *bw = (THDoubleTensor*)luaT_checkudata(L, 4, "torch.DoubleTensor");
    
    double *gradInput_data = THDoubleTensor_data(gradInput);
    double *gradIm2col_data = THDoubleTensor_data(gradIm2col);
    long *bi_data = THLongTensor_data(bi);
    double *bw_data = THDoubleTensor_data(bw);
    
    std::size_t H = THDoubleTensor_size(gradIm2col, 0);
    std::size_t W = THDoubleTensor_size(gradIm2col, 1);
    
    std::size_t s0 = THDoubleTensor_stride(gradIm2col, 0);
    std::size_t s1 = THDoubleTensor_stride(gradIm2col, 1);

    std::size_t t0 = THDoubleTensor_stride(gradInput, 0);
    std::size_t t1 = THDoubleTensor_stride(gradInput, 1);
    std::size_t t2 = THDoubleTensor_stride(gradInput, 2);

    
    std::size_t r0 = THDoubleTensor_stride(bw, 0);
    std::size_t r1 = THDoubleTensor_stride(bw, 1);
    std::size_t r2 = THDoubleTensor_stride(bw, 2);
    
    std::size_t q0 = THLongTensor_stride(bi, 0);
    std::size_t q1 = THLongTensor_stride(bi, 1);
    std::size_t q2 = THLongTensor_stride(bi, 2);
    
    std::size_t i,j;
    
    
    for(i=0; i<H; i++){
       for(j=0; j<W; j++){
           long c = bi_data[q0*i+q1*j+q2*0];
           long ay = bi_data[q0*i+q1*j+q2*1];
           long ax = bi_data[q0*i+q1*j+q2*2];
           
           gradInput_data[c*t0+ay*t1+ax*t2]+= 
                    bw_data[r0*i+r1*j+r2*0]*gradIm2col_data[s0*i+s1*j];
           gradInput_data[c*t0+ay*t1+(ax+1)*t2]+= 
                    bw_data[r0*i+r1*j+r2*1]*gradIm2col_data[s0*i+s1*j];
           gradInput_data[c*t0+(ay+1)*t1+ax*t2]+= 
                    bw_data[r0*i+r1*j+r2*2]*gradIm2col_data[s0*i+s1*j];
           gradInput_data[c*t0+(ay+1)*t1+(ax+1)*t2]+= 
                    bw_data[r0*i+r1*j+r2*3]*gradIm2col_data[s0*i+s1*j];
       }
    }
    
    // We made a copy of the LUA object pointing to the tensor data, so increase the reference counter.
    THDoubleTensor_retain(gradInput);
    luaT_pushudata(L, gradInput, "torch.DoubleTensor");
    return 1;
}

    


static const struct luaL_Reg Global_funcs[] = {
    {"im2colold", im2colold},
	{"im2col", im2col},
    {"bilinearInterpolation", bilinearInterpolation},
    {"update_grad_input", update_grad_input},
	{NULL, NULL}
};

extern "C"
{

DLL_EXPORT int luaopen_libdeformableconvolution(lua_State* L)
{
	lua_newtable(L);
	luaT_setfuncs(L, Global_funcs, 0);
	lua_setglobal(L, "deformableconvolution");

	return 0;
}

}
