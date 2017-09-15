
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
static double bilinearInterp(THDoubleTensor* input, std::size_t c, double py, double px){    
    std::size_t H = THDoubleTensor_size(input, 1); // input image height
    std::size_t W = THDoubleTensor_size(input, 2); // input image width
    
    double *data = THDoubleTensor_data(input);
    
    std::size_t s0 = THDoubleTensor_stride(input, 0);
    std::size_t s1 = THDoubleTensor_stride(input, 1);
    std::size_t s2 = THDoubleTensor_stride(input, 2);
    
    double retval = 0;
    
    if(px <= -1 or px >= W or py <= -1 or py >= H){
        return retval;
    }
    
    double px_projected = projection(px, W-1);
    double py_projected = projection(py, H-1);
    
    int ax = int(floor(px_projected)); 
    int ay = int(floor(py_projected));
    
    if(ax == W-1)
        --ax;
    if(ay == H-1)
        --ay;
    
    retval += (1-std::abs(py_projected - ay))*(1-std::abs(px_projected-ax))*data[s0*c+s1*ay+s2*ax];
    retval += (1-std::abs(py_projected - (ay+1)))*(1-std::abs(px_projected-ax))*data[s0*c+s1*(ay+1)+s2*ax];
    retval += (1-std::abs(py_projected - ay))*(1-std::abs(px_projected-(ax+1)))*data[s0*c+s1*ay+s2*(ax+1)];
    retval += (1-std::abs(py_projected - (ay+1)))*(1-std::abs(px_projected-(ax+1)))*data[s0*c+s1*(ay+1)+s2*(ax+1)];

    if(px_projected != px or py_projected != py){
        retval *= (1-std::abs(px-px_projected))*(1-std::abs(py-py_projected));
    }
        
    return retval;
}

/* @brief lua-wrapper for the function bilinearInterp
 * @param L lua_State which contains the parameters 3D-input-image, input channel, interpolation point parameters py, px
 * @return 1 hands the bilinear Interpolation retval of image(c,py,px) back to lua_State
 */
static int bilinearInterpolation(lua_State *L){
    THDoubleTensor *image = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
    std::size_t c = luaL_checkinteger(L, 2)-1; 
    double py = luaL_checknumber(L, 3)-1;
    double px = luaL_checknumber(L, 4)-1;
    
    double retval = bilinearInterp(image, c, py, px);
    
    lua_pushnumber(L, retval);
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
    THDoubleTensor *offsets = (THDoubleTensor*)luaT_checkudata(L, 2, "torch.DoubleTensor"); //dimension(offsets) = C x kH x kW x H_out x W_out x 2
    std::size_t kH = luaL_checkinteger(L, 3); //kernel height
    std::size_t kW = luaL_checkinteger(L, 4); //kernel width
    std::size_t C = THDoubleTensor_size(image, 0); //number of input channels
    std::size_t H = THDoubleTensor_size(image, 1); //input image height
    std::size_t W = THDoubleTensor_size(image, 2); //input image width
    
    std::size_t wOutputImage = W-kW+1;
    std::size_t hOutputImage = H-kH+1;
    THDoubleTensor* outputImage = THDoubleTensor_newWithSize2d(kW*kH*C, wOutputImage*hOutputImage); //im2col(input)
    
    std::size_t s0 = THDoubleTensor_stride(outputImage, 0);
    std::size_t s1 = THDoubleTensor_stride(outputImage, 1);
    std::size_t t0 = THDoubleTensor_stride(offsets, 0);
    std::size_t t1 = THDoubleTensor_stride(offsets, 1);
    std::size_t t2 = THDoubleTensor_stride(offsets, 2);
    std::size_t t3 = THDoubleTensor_stride(offsets, 3);
    std::size_t t4 = THDoubleTensor_stride(offsets, 4);
    std::size_t t5 = THDoubleTensor_stride(offsets, 5);
    std::size_t c,k,l,i,j;
    
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
                        x_offset = offsets_data[t0*c+t1*l+t2*k+t3*i+t4*j+t5*0];
                        y_offset = offsets_data[t0*c+t1*l+t2*k+t3*i+t4*j+t5*1];
                        odata[(c*kW*kH+l*kW+k)*s0+s1*(i*wOutputImage+j)] = bilinearInterp(image, c, i+l+y_offset, j+k+x_offset);
                    }
                }
            }
        }
    }
    luaT_pushudata(L, outputImage, "torch.DoubleTensor");

    return 1;
}

/* @brief rotates the 2D-matrix input[c2][c1] for 180 degrees for every pair (c2,c1)
 * @param L lua_State which contains the 4D-input-matrix
 * @return 1 hands the rotated input matrix rotatedInput back to the lua_State
 */
static int rotate(lua_State *L){
    THDoubleTensor *input = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor"); //4D-input
    std::size_t c2 = THDoubleTensor_size(input, 0);
    std::size_t c1 = THDoubleTensor_size(input, 1);
    std::size_t h = THDoubleTensor_size(input, 2);
    std::size_t w = THDoubleTensor_size(input, 3);
    
    THDoubleTensor* inputRotated = THDoubleTensor_newWithSize4d(c2, c1, h, w);
    double * idata = THDoubleTensor_data(input);
    double * odata = THDoubleTensor_data(inputRotated);
    
    std::size_t s0 = THDoubleTensor_stride(input, 0);
    std::size_t s1 = THDoubleTensor_stride(input, 1);
    std::size_t s2 = THDoubleTensor_stride(input, 2);
    std::size_t s3 = THDoubleTensor_stride(input, 3);
    
    std::size_t c1i, c2i, i, j;
    
    for(c2i = 0; c2i < c2; c2i++){
        for(c1i = 0; c1i < c1; c1i++){
            for(i = 0; i < h; i++){
                for(j = 0; j < w; j++){
                    odata[s0 * c2i + s1 * c1i + s2 * i + s3 * j] = idata[s0 * c2i + s1 * c1i + +s2 * (h-i-1) + s3 * (w-j-1)];   
                }   
            }
        }
    }
    
    luaT_pushudata(L, inputRotated, "torch.DoubleTensor"); 
    return 1;
}

/* @brief frames the 2D-matrix input[c2] for every c2 with padH zeros in both vertical directions
 * and with padW zeros in both horizontal directions
 * @param L lua_State which contains the 3D-input-matrix and padding sizes padH, padW
 * @return 1 hands the framed matrix finput back to the lua_State
 */
static int frame(lua_State *L){
    THDoubleTensor *input = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor"); //3D-input
    std::size_t padH = luaL_checkinteger(L, 2); //vertical padding
    std::size_t padW = luaL_checkinteger(L, 3); //horizontal padding
    
    std::size_t C2 = THDoubleTensor_size(input, 0);
    std::size_t h = THDoubleTensor_size(input, 1);
    std::size_t w = THDoubleTensor_size(input, 2);
    
    THDoubleTensor* finput = THDoubleTensor_newWithSize3d(C2, h + 2*padH, w + 2*padW);
    
    std::size_t s0 = THDoubleTensor_stride(input, 0);
    std::size_t s1 = THDoubleTensor_stride(input, 1);
    std::size_t s2 = THDoubleTensor_stride(input, 2);
    
    std::size_t t0 = THDoubleTensor_stride(finput, 0);
    std::size_t t1 = THDoubleTensor_stride(finput, 1);
    std::size_t t2 = THDoubleTensor_stride(finput, 2);
    
    double * idata = THDoubleTensor_data(input);
    double * odata = THDoubleTensor_data(finput);
    
    std::size_t c2,i,j;
    for(c2 = 0; c2 < C2; c2++){
        
        for(i = 0; i < padH; i++){
            for(j = 0; j < w+2*padW; j++){
                odata[c2*t0 + i*t1 + j*t2]=0;
                odata[c2*t0 + (i+h+padH)*t1 + j*t2]=0;
            }
        }
        
        for(i = padH; i < h+2*padH; i++){
            for(j = 0; j < padW; j++){
                odata[c2*t0 + i*t1 + j*t2]=0;
                odata[c2*t0 + i*t1 + (j+w+padW)*t2]=0;
            }
        }
        
        for(i = padH; i < h+padH; i++){
            for(j = padW; j < w+padW; j++){
                odata[c2*t0 + i*t1 + j*t2]=idata[c2*s0 + (i-padH)*s1 + (j-padW)*s2];
            }
        }
    }
    
    luaT_pushudata(L, finput, "torch.DoubleTensor"); 
    return 1;
    
}

static int update_grad_input(lua_State *L)
    self.gradInput = torch.Tensor(input:size()):zero()
    for c1star = 1,self.nInputPlane do
        for istar = 1, input:size(2) do
            for jstar = 1,input:size(3) do
                for i = 1,gradOutput:size(2) do
                    for j = 1,gradOutput:size(3) do 
                        for c2 =1,self.nOutputPlane do
                            k = istar + 1 - i
                            l = jstar + 1 - j
                            if(1 <= l and l <= self.kW and 1 <= k and k <= self.kH) then
                                self.gradInput[c1star][istar][jstar] = self.gradInput[c1star][istar][jstar] + gradOutput[c2][i][j]*self.weight[c2][c1star][k][l]
                            end
                        end
                    end
                end
            end
        end
    end
    return self.gradInput
end
    
    


static const struct luaL_Reg Global_funcs[] = {
	{"im2col", im2col},
    {"rotate", rotate},
    {"frame", frame},
    {"bilinearInterpolation", bilinearInterpolation},
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
